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
            # Create gradient parameters on CPU to avoid DTensor issues
            grad_param = torch.nn.Parameter(
                torch.zeros_like(p, device='cpu'), 
                requires_grad=False
            )
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
                n = p.numel()
                slice_ = delta_gpu[offset:offset + n].view_as(p).to(p.dtype)
                # Ensure gradient parameter has the right dtype and device
                if g.device != p.device or g.dtype != p.dtype:
                    g.data = torch.zeros_like(p)
                # Try to copy safely, handling potential DTensor issues
                try:
                    g.data.copy_(slice_)
                except RuntimeError as e:
                    if "DTensor" in str(e):
                        # Handle DTensor case by assigning directly
                        g.data = slice_.detach().clone()
                    else:
                        raise e
                # Ensure the gradient is on the same device as the parameter
                p.grad = g.data.to(p.device)
                offset += n
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
