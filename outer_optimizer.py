from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class OuterState:
    optimizer: dict
    prev_vector: torch.Tensor
    momentum_buffer: Optional[torch.Tensor]


class OuterOptimizer:
    """Helper handling the DiLoCo outer optimization step."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        device_mesh=None,
        momentum: float = 0.0,
        grad_clip: Optional[float] = None,
        lr_scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.device_mesh = device_mesh
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum

        with torch.no_grad():
            params = [p.detach().cpu().to(torch.float16) for p in model.parameters()]
            self.prev_vector = torch.nn.utils.parameters_to_vector(params)
        if momentum > 0:
            self.momentum_buffer = torch.zeros_like(self.prev_vector)
        else:
            self.momentum_buffer = None

        self.grad_params = [
            torch.nn.Parameter(torch.zeros_like(p), requires_grad=False)
            for p in model.parameters()
        ]

    # ------------------------------------------------------------------
    def state_dict(self) -> OuterState:
        return OuterState(
            optimizer=self.optimizer.state_dict(),
            prev_vector=self.prev_vector.clone(),
            momentum_buffer=self.momentum_buffer.clone() if self.momentum_buffer is not None else None,
        )

    # ------------------------------------------------------------------
    def load_state_dict(self, state: OuterState) -> None:
        self.optimizer.load_state_dict(state.optimizer)
        self.prev_vector.copy_(state.prev_vector)
        if self.momentum_buffer is not None and state.momentum_buffer is not None:
            self.momentum_buffer.copy_(state.momentum_buffer)

    # ------------------------------------------------------------------
    def step(self) -> float:
        if self.device_mesh is not None:
            dist.barrier()
        with torch.no_grad():
            curr_vector = torch.nn.utils.parameters_to_vector(
                [p.detach().cpu() for p in self.model.parameters()]
            ).to(self.prev_vector.dtype)
            delta = curr_vector - self.prev_vector
            if self.momentum_buffer is not None:
                self.momentum_buffer.mul_(self.momentum).add_(delta)
                delta = self.momentum_buffer
            delta_gpu = delta.to(self.device, dtype=self.grad_params[0].dtype)
            handle = None
            if self.device_mesh is not None:
                pg = self.device_mesh.get_group()
                handle = dist.all_reduce(delta_gpu, group=pg, async_op=True)
            if handle is not None:
                handle.wait()
            torch.nn.utils.vector_to_parameters(delta_gpu, self.grad_params)
            for p, g in zip(self.model.parameters(), self.grad_params):
                p.grad = g.data
            self.prev_vector.copy_(curr_vector)
        delta_norm = delta.norm().item()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        for p in self.model.parameters():
            p.grad = None
        return delta_norm
