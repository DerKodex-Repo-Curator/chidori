import pytest
import torch
import torch.nn as nn
import time
import os
import numpy as np

# Test configuration
PARTIAL_OFFLOAD_ENABLED = os.environ.get('PARTIAL_OFFLOAD_MODE', '0') == '1'


class SimpleNet(nn.Module):
    """Simple neural network for testing compute on both devices."""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestDeviceEnumeration:
    """Tests for device enumeration and visibility."""
    
    def test_pytorch_sees_two_devices(self):
        """PyTorch must see exactly 2 CUDA devices in partial offload mode."""
        assert torch.cuda.is_available(), "CUDA must be available"
        device_count = torch.cuda.device_count()
        assert device_count == 2, f"Expected 2 CUDA devices, found {device_count}"
    
    def test_device_properties_accessible(self):
        """Device properties must be accessible for both devices."""
        for i in range(2):
            props = torch.cuda.get_device_properties(i)
            assert props.name is not None, f"Device {i} name is None"
            assert props.total_memory > 0, f"Device {i} has no memory reported"


class TestMemoryOperations:
    """Tests for memory allocation and transfers."""
    
    def test_allocate_on_cuda0(self):
        """Memory allocation on cuda:0 (local GPU) must work."""
        t = torch.randn(1000, 1000, device='cuda:0')
        assert t.device.index == 0
        assert t.sum().item() != 0  # Verify data is accessible
    
    def test_allocate_on_cuda1(self):
        """Memory allocation on cuda:1 (remote GPU) must work."""
        t = torch.randn(1000, 1000, device='cuda:1')
        assert t.device.index == 1
        assert t.sum().item() != 0
    
    def test_host_to_device_cuda0(self):
        """CPU-to-GPU transfer to cuda:0 must work."""
        host_data = torch.randn(500, 500)
        device_data = host_data.to('cuda:0')
        result = device_data.to('cpu')
        assert torch.allclose(host_data, result, atol=1e-5)
    
    def test_host_to_device_cuda1(self):
        """CPU-to-GPU transfer to cuda:1 must work."""
        host_data = torch.randn(500, 500)
        device_data = host_data.to('cuda:1')
        result = device_data.to('cpu')
        assert torch.allclose(host_data, result, atol=1e-5)
    
    def test_device_to_host_cuda0(self):
        """GPU-to-CPU transfer from cuda:0 must work."""
        device_data = torch.randn(500, 500, device='cuda:0')
        host_data = device_data.to('cpu')
        assert host_data.device.type == 'cpu'
    
    def test_device_to_host_cuda1(self):
        """GPU-to-CPU transfer from cuda:1 must work."""
        device_data = torch.randn(500, 500, device='cuda:1')
        host_data = device_data.to('cpu')
        assert host_data.device.type == 'cpu'


class TestDeviceToDeviceTransfers:
    """Tests for GPU-to-GPU transfers between cuda:0 and cuda:1."""
    
    def test_cuda0_to_cuda1_transfer(self):
        """cuMemcpyDtoD from cuda:0 to cuda:1 must work."""
        source = torch.randn(100, 100, device='cuda:0')
        dest = source.to('cuda:1')
        assert dest.device.index == 1
        assert torch.allclose(source.to('cpu'), dest.to('cpu'), atol=1e-4)
    
    def test_cuda1_to_cuda0_transfer(self):
        """cuMemcpyDtoD from cuda:1 to cuda:0 must work."""
        source = torch.randn(100, 100, device='cuda:1')
        dest = source.to('cuda:0')
        assert dest.device.index == 0
        assert torch.allclose(source.to('cpu'), dest.to('cpu'), atol=1e-4)
    
    def test_large_transfer_cuda0_to_cuda1(self):
        """Large GPU-to-GPU transfers must complete without timeout."""
        # 100MB transfer
        size = 25 * 1024 * 1024  # 25M floats = 100MB
        source = torch.randn(size, device='cuda:0')
        dest = source.to('cuda:1')
        assert dest.sum().item() != 0
    
    def test_async_transfer_cuda0_to_cuda1(self):
        """cuMemcpyDtoDAsync between devices must work."""
        source = torch.randn(1000, 1000, device='cuda:0')
        # PyTorch async transfer
        dest = source.to('cuda:1', non_blocking=True)
        torch.cuda.synchronize()
        assert torch.allclose(source.to('cpu'), dest.to('cpu'), atol=1e-4)


class TestComputeOperations:
    """Tests for compute operations on both devices."""
    
    def test_kernel_launch_cuda0(self):
        """Kernel launches on cuda:0 must not crash."""
        a = torch.randn(100, 100, device='cuda:0')
        b = torch.randn(100, 100, device='cuda:0')
        c = torch.matmul(a, b)
        assert c.shape == (100, 100)
        assert not torch.isnan(c).any()
    
    def test_kernel_launch_cuda1(self):
        """Kernel launches on cuda:1 must not crash."""
        a = torch.randn(100, 100, device='cuda:1')
        b = torch.randn(100, 100, device='cuda:1')
        c = torch.matmul(a, b)
        assert c.shape == (100, 100)
        assert not torch.isnan(c).any()
    
    def test_cublas_operation_cuda0(self):
        """cuBLAS operations on cuda:0 must work."""
        a = torch.randn(512, 512, device='cuda:0')
        b = torch.randn(512, 512, device='cuda:0')
        c = torch.mm(a, b)
        result = c.sum().item()
        assert not np.isnan(result)
    
    def test_cublas_operation_cuda1(self):
        """cuBLAS operations on cuda:1 must work."""
        a = torch.randn(512, 512, device='cuda:1')
        b = torch.randn(512, 512, device='cuda:1')
        c = torch.mm(a, b)
        result = c.sum().item()
        assert not np.isnan(result)
    
    def test_cudnn_convolution_cuda0(self):
        """cuDNN convolution on cuda:0 must work."""
        conv = nn.Conv2d(3, 16, 3, padding=1).cuda(0)
        x = torch.randn(4, 3, 32, 32, device='cuda:0')
        out = conv(x)
        assert out.shape == (4, 16, 32, 32)
    
    def test_cudnn_convolution_cuda1(self):
        """cuDNN convolution on cuda:1 must work."""
        conv = nn.Conv2d(3, 16, 3, padding=1).cuda(1)
        x = torch.randn(4, 3, 32, 32, device='cuda:1')
        out = conv(x)
        assert out.shape == (4, 16, 32, 32)
    
    def test_complex_operation_cuda0(self):
        """Complex operations (softmax, layer norm) on cuda:0 must work."""
        x = torch.randn(32, 128, device='cuda:0')
        y = torch.softmax(x, dim=-1)
        z = torch.layer_norm(x, [128])
        assert not torch.isnan(y).any()
        assert not torch.isnan(z).any()
    
    def test_complex_operation_cuda1(self):
        """Complex operations (softmax, layer norm) on cuda:1 must work."""
        x = torch.randn(32, 128, device='cuda:1')
        y = torch.softmax(x, dim=-1)
        z = torch.layer_norm(x, [128])
        assert not torch.isnan(y).any()
        assert not torch.isnan(z).any()


class TestCrossDeviceCompute:
    """Tests for operations spanning both devices."""
    
    def test_model_split_across_devices(self):
        """Model layers split across cuda:0 and cuda:1 must work."""
        # First layer on cuda:0
        layer1 = nn.Linear(128, 64).cuda(0)
        # Second layer on cuda:1
        layer2 = nn.Linear(64, 10).cuda(1)
        
        x = torch.randn(8, 128, device='cuda:0')
        h = layer1(x)
        h = h.to('cuda:1')  # Cross-device transfer
        out = layer2(h)
        
        assert out.shape == (8, 10)
        assert not torch.isnan(out).any()
    
    def test_backward_pass_cross_device(self):
        """Backward pass across devices must compute gradients."""
        layer1 = nn.Linear(64, 32).cuda(0)
        layer2 = nn.Linear(32, 10).cuda(1)
        
        x = torch.randn(4, 64, device='cuda:0')
        y = torch.randint(0, 10, (4,), device='cuda:1')
        
        h = layer1(x)
        h = h.to('cuda:1')
        out = layer2(h)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        
        assert layer1.weight.grad is not None
        assert layer2.weight.grad is not None


class TestEndToEndTraining:
    """End-to-end training tests with partial offloading."""
    
    def test_simple_training_loop(self):
        """Simple training loop must complete without crashes."""
        model = SimpleNet().cuda(0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(10):
            x = torch.randn(16, 128, device='cuda:0')
            y = torch.randint(0, 10, (16,), device='cuda:0')
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        assert loss.item() < 100  # Loss should be finite
    
    def test_split_model_training(self):
        """Training with model split across devices must work."""
        part1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU()).cuda(0)
        part2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)).cuda(1)
        
        params = list(part1.parameters()) + list(part2.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(5):
            x = torch.randn(8, 128, device='cuda:0')
            y = torch.randint(0, 10, (8,), device='cuda:1')
            
            optimizer.zero_grad()
            h = part1(x)
            h = h.to('cuda:1')
            out = part2(h)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        assert loss.item() < 100


class TestInference:
    """Inference tests with partial offloading."""
    
    def test_inference_cuda0(self):
        """Inference on cuda:0 must produce valid outputs."""
        model = SimpleNet().cuda(0)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 128, device='cuda:0')
            out = model(x)
        
        assert out.shape == (1, 10)
        assert not torch.isnan(out).any()
    
    def test_inference_cuda1(self):
        """Inference on cuda:1 must produce valid outputs."""
        model = SimpleNet().cuda(1)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 128, device='cuda:1')
            out = model(x)
        
        assert out.shape == (1, 10)
        assert not torch.isnan(out).any()
    
    def test_batch_inference_healthy(self):
        """Batch inference on both devices must work correctly."""
        model = SimpleNet()
        
        # Test on cuda:0
        model_cuda0 = model.cuda(0)
        model_cuda0.eval()
        x0 = torch.randn(32, 128, device='cuda:0')
        with torch.no_grad():
            out0 = model_cuda0(x0)
        assert out0.shape == (32, 10)
        
        # Test on cuda:1
        model_cuda1 = model.cuda(1)
        model_cuda1.eval()
        x1 = torch.randn(32, 128, device='cuda:1')
        with torch.no_grad():
            out1 = model_cuda1(x1)
        assert out1.shape == (32, 10)


class TestRegressionSingleServer:
    """Tests to ensure no regressions in single-server (non-partial) mode."""
    
    @pytest.mark.skipif(PARTIAL_OFFLOAD_ENABLED, reason="Only for non-partial mode")
    def test_single_server_basic_compute(self):
        """Basic compute must still work in single-server mode."""
        t = torch.randn(100, 100, device='cuda:0')
        result = torch.matmul(t, t)
        assert result.shape == (100, 100)


class TestGetProcAddressHandling:
    """Tests for cuGetProcAddress forwarding that was causing crashes."""
    
    def test_cuda_runtime_init_cuda0(self):
        """CUDA runtime initialization on cuda:0 must not crash."""
        # This implicitly tests cuGetProcAddress for runtime functions
        torch.cuda.init()
        t = torch.randn(10, 10, device='cuda:0')
        assert t is not None
    
    def test_cuda_runtime_init_cuda1(self):
        """CUDA runtime initialization on cuda:1 must not crash."""
        torch.cuda.init()
        t = torch.randn(10, 10, device='cuda:1')
        assert t is not None
    
    def test_mixed_device_operations(self):
        """Rapid switching between devices must not cause crashes."""
        for _ in range(20):
            a = torch.randn(50, 50, device='cuda:0')
            b = torch.randn(50, 50, device='cuda:1')
            c = a.sum() + b.sum().to('cuda:0')
        assert c is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])