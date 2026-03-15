/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 *  Modified 2026 — removed Vulkan/version deps, CUDA-only LD_PRELOAD.
 */
package app

import (
	"fmt"
	"os/exec"
	"path/filepath"

	"github.com/fpolica91/juice/control-plane/pkg/task"
)

func getVersion() (string, error) {
	return "2.0.0-oss", nil
}

func validateHost() error {
	return nil
}

func createCommand(args []string) *exec.Cmd {
	return exec.Command(args[0], args[1:]...)
}

func runCommand(group task.Group, cmd *exec.Cmd, config Configuration) error {
	juiceLibraryPath := filepath.Join(*juicePath, "libjuice-cuda.so")
	cmd.Env = append(cmd.Env, fmt.Sprintf("LD_PRELOAD=%s", juiceLibraryPath))
	return cmd.Run()
}
