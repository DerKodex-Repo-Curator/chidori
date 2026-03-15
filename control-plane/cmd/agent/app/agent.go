/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 */
package app

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"

	cmdgpu "github.com/fpolica91/chidori/control-plane/cmd/agent/gpu"
	"github.com/fpolica91/chidori/control-plane/pkg/errors"
	"github.com/fpolica91/chidori/control-plane/pkg/gpu"
	"github.com/fpolica91/chidori/control-plane/pkg/logger"
	"github.com/fpolica91/chidori/control-plane/pkg/restapi"
	"github.com/fpolica91/chidori/control-plane/pkg/server"
	"github.com/fpolica91/chidori/control-plane/pkg/task"
	"github.com/fpolica91/chidori/control-plane/pkg/utilities"
)

var (
	binPath = flag.String("bin-path", "", "")

	address = flag.String("address", "0.0.0.0:43210", "The IP address and port to use for listening for client connections")
	labels  = flag.String("labels", "", "Comma separated list of key=value pairs")
	taints  = flag.String("taints", "", "Comma separated list of key=value pairs")
	poolId  = flag.String("pool-id", "", "The ID of the pool this agent belongs to")
)

type EventListener interface {
	SessionClosed(id string)

	ConnectionCreated(sessionId string, connection restapi.ConnectionData)
	ConnectionClosed(sessionId string, connection restapi.ConnectionData, exitCode int)
}

type Agent struct {
	Id string

	Hostname  string
	BinPath string

	Gpus               *gpu.GpuSet
	GpuMetricsProvider *cmdgpu.MetricsProvider

	Server *server.Server

	labels map[string]string
	taints map[string]string
	poolId string

	sessions    *utilities.ConcurrentMap[string, *Session]
	taskManager *task.TaskManager

	controllerData
}

func NewAgent(ctx context.Context, tlsConfig *tls.Config) (*Agent, error) {
	if tlsConfig == nil {
		logger.Warning("TLS is disabled, data will be unencrypted")
	}

	server, err := server.NewServer(*address, tlsConfig)
	if err != nil {
		return nil, errors.New("failed to create server").Wrap(err)
	}

	agent := &Agent{
		Id:          uuid.NewString(),
		BinPath:   *binPath,
		Server:      server,
		labels:      map[string]string{},
		taints:      map[string]string{},
		sessions:    utilities.NewConcurrentMap[string, *Session](),
		taskManager: task.NewTaskManager(ctx),
	}

	if *labels != "" {
		var err error
		for _, tag := range strings.Split(*labels, ",") {
			keyValue := strings.Split(tag, "=")
			if len(keyValue) != 2 {
				err = errors.Join(err, errors.Newf("tag '%s' must be in the format key=value", tag))
			} else {
				agent.labels[strings.TrimSpace(keyValue[0])] = strings.TrimSpace(keyValue[1])
			}
		}

		if err != nil {
			return nil, errors.New("failed to parse --labels").Wrap(err)
		}
	}

	if *taints != "" {
		var err error
		for _, taint := range strings.Split(*taints, ",") {
			keyValue := strings.Split(taint, "=")
			if len(keyValue) != 2 {
				err = errors.Join(err, errors.Newf("taint '%s' must be in the format key=value", taint))
			} else {
				agent.taints[strings.TrimSpace(keyValue[0])] = strings.TrimSpace(keyValue[1])
			}
		}

		if err != nil {
			return nil, errors.New("failed to parse --taints").Wrap(err)
		}
	}

	if *poolId != "" {
		agent.poolId = *poolId
	}

	if agent.BinPath == "" {
		executable, err := os.Executable()
		if err != nil {
			return nil, err
		}

		agent.BinPath = filepath.Dir(executable)
	}

	hostname, err := os.Hostname()
	if err != nil {
		return nil, errors.New("failed to retrieve system hostname").Wrap(err)
	}

	agent.Hostname = hostname

	agent.Gpus, err = cmdgpu.DetectGpus()
	if err != nil {
		return nil, errors.New("failed to detect GPUs").Wrap(err)
	}

	logger.Info("GPUs")
	for _, gpuInfo := range agent.Gpus.GetGpus() {
		logger.Infof("  %d @ %s: %s %dMB", gpuInfo.Index, gpuInfo.PciBus, gpuInfo.Name, gpuInfo.Vram/(1024*1024))
	}

	agent.GpuMetricsProvider = cmdgpu.NewMetricsProvider(agent.Gpus)

	agent.initializeEndpoints()

	return agent, nil
}

func (agent *Agent) Run(group task.Group) error {
	logger.Infof("Starting agent on %s", *address)

	group.Go("Agent GpuMetricsProvider", agent.GpuMetricsProvider)
	group.Go("Agent Server", agent.Server)

	return agent.taskManager.Wait()
}

func (agent *Agent) getSession(sessionId string) (*Session, error) {
	session, found := agent.sessions.Get(sessionId)
	if !found {
		return nil, errors.Newf("session with id %s not found", sessionId)
	}

	return session, nil
}

func (agent *Agent) addSession(sessionId string, version string, gpus *gpu.SelectedGpuSet) {
	logger.Debugf("Starting Session %s", sessionId)

	session := newSession(agent.taskManager.Ctx(), sessionId, version, agent.BinPath, gpus, agent)
	agent.sessions.Set(sessionId, session)

	agent.taskManager.Go(fmt.Sprintf("session %s", sessionId), session)

	agent.SessionActive(sessionId)
}

func (agent *Agent) cancelSession(sessionId string) error {
	session, err := agent.getSession(sessionId)
	if err == nil {
		session.Cancel()
	}

	if err != nil {
		err = errors.New("unable to cancel session").Wrap(err)
	}

	return err
}

func (agent *Agent) connect(sessionId string, connectionData restapi.ConnectionData, c net.Conn) error {
	session, err := agent.getSession(sessionId)
	if err == nil {
		err = session.Connect(connectionData, c)
	}

	if err != nil {
		err = errors.New("unable to connect to session").Wrap(err)
	}

	return err
}

func (agent *Agent) requestSession(sessionRequirements restapi.SessionRequirements) (string, error) {
	selectedGpus, err := agent.Gpus.Find(sessionRequirements.Gpus)
	if err != nil {
		return "", errors.New("unable to find a matching set of GPUs").Wrap(err)
	}

	id := uuid.NewString()
	agent.addSession(id, sessionRequirements.Version, selectedGpus)
	return id, nil
}

func (agent *Agent) registerSession(session restapi.Session) error {
	selectedGpus, err := agent.Gpus.Select(session.Gpus)
	if err != nil {
		return errors.New("unable to select a matching set of GPUs").Wrap(err)
	}

	agent.addSession(session.Id, session.Version, selectedGpus)
	return nil
}

func (agent *Agent) SessionActive(id string) {
	logger.Debugf("session %s active", id)

	if agent.sessionUpdates != nil {
		agent.sessionUpdates <- sessionUpdate{
			Id:    id,
			State: restapi.SessionActive,
		}
	}
}

func (agent *Agent) SessionClosed(id string) {
	logger.Debugf("session %s closed", id)

	agent.sessions.Delete(id)

	if agent.sessionUpdates != nil {
		agent.sessionUpdates <- sessionUpdate{
			Id:    id,
			State: restapi.SessionClosed,
		}
	}
}

func (agent *Agent) ConnectionCreated(sessionId string, connection restapi.ConnectionData) {
	logger.Debugf("session %s created connection %s", sessionId, connection.Id)

	if agent.connectionUpdates != nil {
		agent.connectionUpdates <- connectionUpdate{
			SessionId: sessionId,
			Connection: restapi.Connection{
				ConnectionData: connection,
			},
		}
	}
}

func (agent *Agent) ConnectionClosed(sessionId string, connection restapi.ConnectionData, exitCode int) {
	logger.Debugf("session %s closed connection %s with exit code %d", sessionId, connection.Id, exitCode)

	if agent.connectionUpdates != nil {
		agent.connectionUpdates <- connectionUpdate{
			SessionId: sessionId,
			Connection: restapi.Connection{
				ConnectionData: connection,
				ExitCode:       exitCode,
			},
		}
	}
}
