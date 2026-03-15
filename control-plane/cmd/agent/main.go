/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 */
package main

import (
	"crypto/tls"
	"flag"
	"fmt"
	"os"

	"github.com/fpolica91/juice/control-plane/cmd/agent/app"
	"github.com/fpolica91/juice/control-plane/cmd/agent/prometheus"
	"github.com/fpolica91/juice/control-plane/cmd/internal/build"
	"github.com/fpolica91/juice/control-plane/pkg/appmain"
	"github.com/fpolica91/juice/control-plane/pkg/crypto"
	"github.com/fpolica91/juice/control-plane/pkg/logger"
	"github.com/fpolica91/juice/control-plane/pkg/sentry"
	"github.com/fpolica91/juice/control-plane/pkg/task"
	"github.com/joho/godotenv"
)

var (
	certFile     = flag.String("cert-file", "", "")
	keyFile      = flag.String("key-file", "", "")
	generateCert = flag.Bool("generate-cert", false, "Generates a certificate for https")
)

func main() {
	name := "Juice Agent"
	config := appmain.Config{
		Name:    name,
		Version: build.Version,

		SentryConfig: sentry.ClientOptions{
			Dsn:              os.Getenv("JUICE_AGENT_SENTRY_DSN"),
			Release:          fmt.Sprintf("%s@%s", name, build.Version),
			EnableTracing:    true,
			TracesSampleRate: 1.0,
		},
	}

	err := appmain.Run(config, func(group task.Group) error {
		var tlsConfig *tls.Config

		if *certFile != "" && *keyFile != "" {
			certificate, err := tls.LoadX509KeyPair(*certFile, *keyFile)
			if err != nil {
				return err
			}

			tlsConfig = &tls.Config{
				InsecureSkipVerify: true,
				Certificates:       []tls.Certificate{certificate},
			}
		} else if *generateCert {
			certificate, err := crypto.GenerateCertificate()
			if err != nil {
				return err
			}

			tlsConfig = &tls.Config{
				InsecureSkipVerify: true,
				Certificates:       []tls.Certificate{certificate},
			}
		}

		if err := godotenv.Load(); err != nil {
			logger.Infof("Could not load .env file: %v", err)
		}

		agent, err := app.NewAgent(group.Ctx(), tlsConfig)
		if err == nil {
			agent.GpuMetricsProvider.AddConsumer(prometheus.NewGpuMetricsConsumer())

			err = agent.ConnectToController(group, tlsConfig)
			if err == nil {
				group.Go("Agent", agent)
			} else {
				group.Cancel()
			}
		}

		return err
	})

	if err != nil {
		os.Exit(appmain.ExitFailure)
	}
}
