import pytest 
from models.layers import LINEAR_REGISTRY
import torch 

@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_instantiation_no_bias(layer_type):
    layer = layer_type(128, 256)
    assert layer is not None
    assert layer.weight is not None
    assert layer.bias is not None


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_instantiation_with_bias(layer_type):
    layer = layer_type(128, 256, bias=True)
    assert layer is not None
    assert layer.weight is not None
    assert layer.bias is not None


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_instantiation_without_bias(layer_type):
    layer = layer_type(128, 256, bias=False)
    assert layer is not None
    assert layer.weight is not None
    assert layer.bias is None


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_forward_no_bias(layer_type):
    layer = layer_type(128, 256)
    x = torch.randn(1, 128) 
    out = layer(x) 
    assert out is not None 
    assert out.shape == (1, 256)
    assert out.dtype == x.dtype


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_forward_with_bias(layer_type):
    layer = layer_type(128, 256, bias=True)
    x = torch.randn(1, 128) 
    out = layer(x) 
    assert out is not None 
    assert out.shape == (1, 256)
    assert out.dtype == x.dtype


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_forward_without_bias(layer_type):
    layer = layer_type(128, 256, bias=False)
    x = torch.randn(1, 128) 
    out = layer(x) 
    assert out is not None 
    assert out.shape == (1, 256)
    assert out.dtype == x.dtype


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_backward_no_bias(layer_type):
    layer = layer_type(128, 256)
    x = torch.randn(1, 128) 
    out = layer(x) 
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    

@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_backward_with_bias(layer_type):
    layer = layer_type(128, 256)
    x = torch.randn(1, 128) 
    out = layer(x) 
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_layer_backward_without_bias(layer_type):
    layer = layer_type(128, 256, bias=False)
    x = torch.randn(1, 128) 
    out = layer(x) 
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert not hasattr(layer, 'bias') or layer.bias is None or layer.bias.grad is None


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_batch_input(layer_type):
    layer = layer_type(128, 256)
    batch_size = 32
    x = torch.randn(batch_size, 128)
    out = layer(x)
    assert out.shape == (batch_size, 256)


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_dtype_preservation(layer_type):
    layer = layer_type(128, 256)
    
    # Test with float32
    x_float32 = torch.randn(1, 128, dtype=torch.float32)
    out_float32 = layer(x_float32)
    assert out_float32.dtype == torch.float32
    
    # Test with float64
    x_float64 = torch.randn(1, 128, dtype=torch.float64)
    layer_float64 = layer_type(128, 256).double()
    out_float64 = layer_float64(x_float64)
    assert out_float64.dtype == torch.float64


@pytest.mark.parametrize("layer_type", LINEAR_REGISTRY.values())
def test_device_placement(layer_type):
    if torch.cuda.is_available():
        layer = layer_type(128, 256).cuda()
        x = torch.randn(1, 128).cuda()
        out = layer(x)
        assert out.is_cuda
        assert out.device == x.device







