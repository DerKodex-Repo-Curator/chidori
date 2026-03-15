/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 */
package app

import (
	"crypto/tls"
	"flag"
	"fmt"
	"io/fs"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/fpolica91/chidori/control-plane/pkg/errors"
	"github.com/fpolica91/chidori/control-plane/pkg/logger"
	"github.com/fpolica91/chidori/control-plane/pkg/restapi"
	"github.com/fpolica91/chidori/control-plane/pkg/task"
	"github.com/fpolica91/chidori/control-plane/pkg/utilities"
)

type Connection struct {
	restapi.ConnectionData

	binPath string
	pciBus    string

	cmd       *exec.Cmd
	readPipe  *os.File
	writePipe *os.File

	// hijackedConn holds a reference to the hijacked TCP connection after
	// it has been forwarded to chidori-worker via SCM_RIGHTS. We MUST
	// retain this reference to prevent Go's garbage collector from
	// finalizing the net.Conn and calling close() on the underlying fd.
	// Even though SCM_RIGHTS creates an independent kernel refcount,
	// Go's Close() on the net.Conn can trigger a TCP RST (e.g. if there
	// is unread data in the receive buffer) which kills the connection
	// for the gpu-worker. The connection is closed when the Connection
	// is torn down.
	hijackedConn net.Conn
}

func newConnection(connectionData restapi.ConnectionData, binPath string, pciBus string) *Connection {
	return &Connection{
		ConnectionData: connectionData,
		binPath:      binPath,
		pciBus:         pciBus,
	}
}

func (connection *Connection) Start(group task.Group, exitCode chan int) error {
	if connection.cmd != nil {
		panic("must call Start() exactly once")
	}

	ch1Read, ch1Write, err := setupIpc()
	if err == nil {
		connection.readPipe = ch1Read
		defer ch1Write.Close()

		ch2Read, ch2Write, err_ := setupIpc()
		err = err_
		if err == nil {
			connection.writePipe = ch2Write
			defer ch2Read.Close()

			logsPath := filepath.Join(connection.binPath, "logs")
			_, err = os.Stat(logsPath)
			if err != nil && os.IsNotExist(err) {
				err = os.MkdirAll(logsPath, fs.ModeDir|fs.ModePerm)
			}

			if err == nil {
				now := time.Now()

				// NOTE: time.Format is really weird. The string below equates to YYYYMMDD-HHMMSS_
				logName := fmt.Sprintf("%s_%s_%s.log", connection.ProcessName, now.Format("20060102-150405"), connection.Id)

				if err == nil {
					connection.cmd = exec.CommandContext(group.Ctx(),
						filepath.Join(connection.binPath, "chidori-worker"),
						append(
							[]string{
								"--id", connection.Id,
								"--log_file", filepath.Join(logsPath, logName),
								"--ipc_write", fmt.Sprint(ch1Write.Fd()),
								"--ipc_read", fmt.Sprint(ch2Read.Fd()),
								"--pcibus", connection.pciBus,
							},
							flag.Args()[0:]...,
						)...,
					)

					inheritFiles(connection.cmd, ch1Write, ch2Read)

					// Capture gpu-worker stderr for debugging
					workerLog, logErr := os.Create(filepath.Join(logsPath, logName+".stderr"))
					if logErr == nil {
						connection.cmd.Stderr = workerLog
						defer workerLog.Close()
					}

					err = connection.cmd.Start()
					if err == nil {
						group.GoFn(fmt.Sprintf("connection %s", connection.Id), func(g task.Group) error {
							waitErr := connection.cmd.Wait()

							// Close IPC pipes and the hijacked TCP connection.
							// The hijacked conn must be closed AFTER the gpu-worker
							// exits so the TCP socket stays alive while it is in use.
							closeErr := errors.Join(
								connection.readPipe.Close(),
								connection.writePipe.Close(),
							)
							if connection.hijackedConn != nil {
								closeErr = errors.Join(closeErr, connection.hijackedConn.Close())
								connection.hijackedConn = nil
							}
							err := errors.Join(waitErr, closeErr)
							if err != nil {
								logger.Errorf("connection %s failed to wait caused by %v", connection.Id, err)
							}

							exitCode <- connection.cmd.ProcessState.ExitCode()

							return nil
						})
					}
				}
			}
		}
	}

	if err != nil {
		err = errors.Join(err,
			connection.readPipe.Close(),
			connection.writePipe.Close())

		err = errors.Newf("connection %s failed to start", connection.Id).Wrap(err)
	}

	return err
}

func (connection *Connection) Connect(c net.Conn) error {
	if connection.cmd == nil {
		panic("must call Start() successfully first")
	}

	tcpConn := &net.TCPConn{}
	tlsConn, err := utilities.Cast[*tls.Conn](c)
	if err == nil {
		tcpConn, err = utilities.Cast[*net.TCPConn](tlsConn.NetConn())
	} else {
		tcpConn, err = utilities.Cast[*net.TCPConn](c)
	}

	if err == nil {
		var rawConn syscall.RawConn
		rawConn, err = tcpConn.SyscallConn()
		if err == nil {
			err = connection.forwardSocket(rawConn)
			if err == nil {
				// Wait for the server to indicate it is ready
				data := make([]byte, 1)
				_, err = connection.readPipe.Read(data)

				// And finally, inform the server to continue
				_, err_ := connection.writePipe.Write(data)
				err = errors.Join(err, err_)
			}
		}
	}

	if err != nil {
		err = errors.Join(err, c.Close())
		err = errors.Newf("connection %s failed to connect", connection.Id).Wrap(err)
	} else {
		// Keep a reference to the hijacked connection so that Go's GC does
		// not finalize it and close the underlying TCP socket fd.  The fd
		// has been forwarded to chidori-worker via SCM_RIGHTS, but the
		// agent's copy must remain open to avoid sending TCP FIN/RST to
		// the remote client.  It will be closed when the Connection is
		// torn down (see the goroutine in Start that waits for cmd.Wait).
		connection.hijackedConn = c
	}

	return err
}
