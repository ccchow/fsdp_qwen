"""Run a quick check that OuterOptimizer handles DTensor shapes correctly."""

import types
import torch
from outer_optimizer import OuterOptimizer


def main() -> None:
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

    outer = OuterOptimizer(model, opt, device=torch.device("cpu"), device_mesh=None)

    with torch.no_grad():
        model.weight.add_(1.0)
    delta_norm = outer.step()

    print("prev_vector elements:", outer.prev_vector.numel())
    print("grad_param shape:", outer.grad_params[0].shape)
    print("weight.grad shape:", model.weight.grad.shape)
    print("delta_norm:", delta_norm)


if __name__ == "__main__":
    main()
