/*
 *  Copyright (c) 2023 Juice Technologies, Inc. All Rights Reserved.
 *  Modified 2026 — replaced Renderer_Win shelling with direct NVML calls.
 */
package gpu

import (
	"fmt"

	"github.com/NVIDIA/go-nvml/pkg/nvml"

	"github.com/fpolica91/juice/control-plane/pkg/gpu"
	"github.com/fpolica91/juice/control-plane/pkg/restapi"
)

func DetectGpus() (*gpu.GpuSet, error) {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("DetectGpus: nvml.Init() failed: %v", nvml.ErrorString(ret))
	}

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("DetectGpus: DeviceGetCount failed: %v", nvml.ErrorString(ret))
	}

	if count == 0 {
		return nil, fmt.Errorf("DetectGpus: no GPUs found")
	}

	apiGpus := make([]restapi.Gpu, 0, count)
	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("DetectGpus: DeviceGetHandleByIndex(%d) failed: %v", i, nvml.ErrorString(ret))
		}

		name, ret := device.GetName()
		if ret != nvml.SUCCESS {
			name = "Unknown"
		}

		uuid, ret := device.GetUUID()
		if ret != nvml.SUCCESS {
			uuid = ""
		}

		memory, ret := device.GetMemoryInfo()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("DetectGpus: GetMemoryInfo(%d) failed: %v", i, nvml.ErrorString(ret))
		}

		pciInfo, ret := device.GetPciInfo()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("DetectGpus: GetPciInfo(%d) failed: %v", i, nvml.ErrorString(ret))
		}

		driverVersion, _ := nvml.SystemGetDriverVersion()

		// Convert PCI bus ID from byte array to string
		pciBus := ""
		for _, b := range pciInfo.BusId {
			if b == 0 {
				break
			}
			pciBus += string(rune(b))
		}

		apiGpus = append(apiGpus, restapi.Gpu{
			Index:    i,
			Uuid:     uuid,
			Name:     name,
			VendorId: uint32(pciInfo.PciDeviceId >> 16),
			DeviceId: uint32(pciInfo.PciDeviceId & 0xFFFF),
			Driver:   driverVersion,
			Vram:     memory.Total,
			PciBus:   pciBus,
		})
	}

	return gpu.NewGpuSet(apiGpus), nil
}
