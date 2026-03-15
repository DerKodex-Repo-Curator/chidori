/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 */
package main

import (
	"fmt"
	"os"

	"github.com/fpolica91/chidori/control-plane/cmd/internal/build"
	"github.com/fpolica91/chidori/control-plane/cmd/juicify/app"
	"github.com/fpolica91/chidori/control-plane/pkg/appmain"
	"github.com/fpolica91/chidori/control-plane/pkg/sentry"
	"github.com/fpolica91/chidori/control-plane/pkg/task"
)

func main() {
	name := "juicify"

	config := appmain.Config{
		Name:    name,
		Version: build.Version,

		SentryConfig: sentry.ClientOptions{
			Dsn:     os.Getenv("JUICE_JUICIFY_SENTRY_DSN"),
			Release: fmt.Sprintf("%s@%s", name, build.Version),
		},
	}

	err := appmain.Run(config, func(group task.Group) error {
		err := app.Run(group)
		group.Cancel()
		return err
	})

	if err != nil {
		os.Exit(appmain.ExitFailure)
	}
}
