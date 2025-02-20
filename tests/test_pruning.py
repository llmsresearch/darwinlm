import pytest
import torch
from darwinlm.pruning.second_order import SecondOrderPruning

@pytest.fixture
def mock_model():
    return torch.nn.Linear(10, 10)

@pytest.fixture
def mock_data():
    return torch.randn(32, 10)

def test_hessian_computation(mock_model, mock_data):
    pruner = SecondOrderPruning(mock_model, mock_data)
    hessian = pruner.compute_hessian(mock_data)
    
    assert hessian.shape == (10, 10)
    assert torch.allclose(hessian, hessian.t())  # Hessian should be symmetric

def test_optimal_mask_computation(mock_model, mock_data):
    pruner = SecondOrderPruning(mock_model, mock_data)
    hessian = pruner.compute_hessian(mock_data)
    weights = mock_model.weight.data
    
    mask, delta = pruner.compute_optimal_mask(weights, hessian, num_dims=5)
    
    assert mask.shape == weights.shape
    assert torch.sum(mask) == 5 * weights.size(1)  # Should keep 5 dimensions 