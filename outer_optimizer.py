from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


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
            # Collect parameters manually to avoid DTensor issues with parameters_to_vector
            param_data = []
            for p in model.parameters():
                # Try to handle DTensor by getting the local part
                try:
                    if hasattr(p, 'to_local'):
                        # DTensor case - get local tensor
                        local_p = p.to_local().detach().cpu().flatten()
                    else:
                        # Regular tensor case
                        local_p = p.detach().cpu().flatten()
                except Exception:
                    # Fallback for any issues
                    local_p = p.detach().cpu().flatten()
                param_data.append(local_p.to(torch.float16))
            
            # Manually concatenate instead of using parameters_to_vector
            self.prev_vector = torch.cat(param_data)
        if momentum > 0:
            self.momentum_buffer = torch.zeros_like(self.prev_vector)
        else:
            self.momentum_buffer = None

        self.grad_params = []
        for p in model.parameters():
            # Create gradient parameters on CPU using the local shard shape
            if hasattr(p, "to_local"):
                local_shape = p.to_local().shape
            else:
                local_shape = p.shape
            grad_tensor = torch.zeros(
                local_shape, dtype=p.dtype, device="cpu"
            )
            grad_param = torch.nn.Parameter(grad_tensor, requires_grad=False)
            self.grad_params.append(grad_param)

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
            params = []
            for p in self.model.parameters():
                # Try to handle DTensor by getting the local part
                try:
                    if hasattr(p, 'to_local'):
                        # DTensor case - get local tensor
                        local_p = p.to_local().detach().cpu()
                    else:
                        # Regular tensor case
                        local_p = p.detach().cpu()
                except Exception:
                    # Fallback for any issues
                    local_p = p.detach().cpu()
                params.append(local_p)
            # Manually concatenate instead of using parameters_to_vector to avoid DTensor issues
            param_data = [p.flatten() for p in params]
            curr_vector = torch.cat(param_data).to(self.prev_vector.dtype)
            delta = curr_vector - self.prev_vector
            if self.momentum_buffer is not None:
                self.momentum_buffer.mul_(self.momentum).add_(delta)
                delta = self.momentum_buffer
            delta_gpu = delta.to(self.device)
            handle = None
            if self.device_mesh is not None:
                pg = self.device_mesh.get_group()
                handle = dist.all_reduce(delta_gpu, group=pg, async_op=True)
            if handle is not None:
                handle.wait()
            offset = 0
            for p, g in zip(self.model.parameters(), self.grad_params):
                local_tensor = p.to_local() if hasattr(p, "to_local") else p
                local_n = local_tensor.numel()
                slice_ = delta_gpu[offset:offset + local_n].view_as(local_tensor).to(p.dtype)
                if g.shape != local_tensor.shape or g.dtype != p.dtype:
                    g.data = torch.zeros(local_tensor.shape, dtype=p.dtype, device="cpu")
                try:
                    g.data.copy_(slice_)
                except RuntimeError as e:
                    if "DTensor" in str(e):
                        g.data = slice_.detach().clone()
                    else:
                        raise e
                if (
                    hasattr(p, "to_local")
                    and hasattr(p, "device_mesh")
                    and hasattr(p.device_mesh, "device_type")
                ):
                    grad = DTensor.from_local(
                        slice_.to(p.device), p.device_mesh, p.placements
                    )
                else:
                    grad = slice_.to(p.device)
                p.grad = grad
                offset += local_n
            self.prev_vector.copy_(curr_vector)
        delta_norm = delta.norm().item()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=False)
        return delta_norm
