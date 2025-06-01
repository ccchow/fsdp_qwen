import types
import torch
from outer_optimizer import OuterOptimizer


def test_outer_optimizer_handles_dtensor_local_size():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    local = model.weight.data
    global_shape = torch.Size([4, 2])
    model.weight.to_local = lambda: local
    model.weight.device_mesh = types.SimpleNamespace(get_group=lambda: None)
    model.weight.placements = (torch.distributed.tensor.Shard(0),)
    model.weight.numel = lambda: global_shape.numel()

    outer = OuterOptimizer(model, opt, device=torch.device('cpu'), device_mesh=None)

    with torch.no_grad():
        model.weight.add_(1.0)
    delta_norm = outer.step()

    assert outer.prev_vector.numel() == local.numel()
    assert outer.grad_params[0].shape == local.shape
    assert model.weight.grad.shape == local.shape
    assert delta_norm > 0
