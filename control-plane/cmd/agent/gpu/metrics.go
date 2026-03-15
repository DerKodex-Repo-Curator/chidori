/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 *  Modified 2026 — replaced Renderer_Win shelling with direct NVML calls.
 */
package gpu

import (
	"flag"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"

	pkggpu "github.com/fpolica91/juice/control-plane/pkg/gpu"
	"github.com/fpolica91/juice/control-plane/pkg/logger"
	"github.com/fpolica91/juice/control-plane/pkg/restapi"
	"github.com/fpolica91/juice/control-plane/pkg/task"
)

var (
	disableGpuMetrics  = flag.Bool("disable-gpu-metrics", false, "")
	gpuMetricsInterval = flag.Uint("gpu-metrics-interval-ms", 1000, "")
)

type MetricsConsumerFn = func([]restapi.Gpu)

type MetricsProvider struct {
	consumers []MetricsConsumerFn
	gpus      *pkggpu.GpuSet
}

func NewMetricsProvider(gpus *pkggpu.GpuSet) *MetricsProvider {
	return &MetricsProvider{
		gpus: gpus,
	}
}

func (provider *MetricsProvider) AddConsumer(consumer MetricsConsumerFn) {
	provider.consumers = append(provider.consumers, consumer)
}

func (provider *MetricsProvider) Run(group task.Group) error {
	if *disableGpuMetrics || len(provider.consumers) == 0 {
		return nil
	}

	ticker := time.NewTicker(time.Duration(*gpuMetricsInterval) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-group.Ctx().Done():
			return nil
		case <-ticker.C:
			gpuList := provider.gpus.GetGpus()
			for i := range gpuList {
				device, ret := nvml.DeviceGetHandleByIndex(i)
				if ret != nvml.SUCCESS {
					logger.Warningf("Failed to get device handle for GPU %d: %v", i, nvml.ErrorString(ret))
					continue
				}

				utilization, ret := device.GetUtilizationRates()
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.UtilizationGpu = utilization.Gpu
					gpuList[i].Metrics.UtilizationVram = utilization.Memory
				}

				memory, ret := device.GetMemoryInfo()
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.VramUsed = memory.Used
				}

				temp, ret := device.GetTemperature(nvml.TEMPERATURE_GPU)
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.TemperatureGpu = temp
				}

				power, ret := device.GetPowerUsage()
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.PowerDraw = power
				}

				powerLimit, ret := device.GetPowerManagementLimit()
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.PowerLimit = powerLimit
				}

				clockCore, ret := device.GetClockInfo(nvml.CLOCK_GRAPHICS)
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.ClockCore = clockCore
				}

				clockMem, ret := device.GetClockInfo(nvml.CLOCK_MEM)
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.ClockMemory = clockMem
				}

				fan, ret := device.GetFanSpeed()
				if ret == nvml.SUCCESS {
					gpuList[i].Metrics.FanSpeed = fan
				}
			}

			for _, consumer := range provider.consumers {
				consumer(gpuList)
			}
		}
	}
}
