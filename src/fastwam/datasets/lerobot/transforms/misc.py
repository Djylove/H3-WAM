from typing import List

import torch


class SelectDimensions:
    def __init__(
        self,
        action_indices: List[int] | None = None,
        state_indices: List[int] | None = None,
        action_key: str = "default",
        state_key: str = "default",
        original_action_dim: int | None = None,
        original_state_dim: int | None = None,
    ):
        self.action_indices = self._normalize_indices(action_indices, original_action_dim, "action_indices")
        self.state_indices = self._normalize_indices(state_indices, original_state_dim, "state_indices")
        self.action_key = action_key
        self.state_key = state_key
        self.original_action_dim = original_action_dim
        self.original_state_dim = original_state_dim

    @staticmethod
    def _normalize_indices(indices: List[int] | None, original_dim: int | None, name: str) -> list[int] | None:
        if indices is None:
            return None
        normalized = [int(idx) for idx in indices]
        if len(normalized) == 0:
            raise ValueError(f"`{name}` must not be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"`{name}` contains duplicate indices: {normalized}.")
        if min(normalized) < 0:
            raise ValueError(f"`{name}` must be non-negative, got {normalized}.")
        if original_dim is not None and max(normalized) >= int(original_dim):
            raise ValueError(f"`{name}` must be < {original_dim}, got {normalized}.")
        return normalized

    @staticmethod
    def _select(x: torch.Tensor, indices: list[int] | None) -> torch.Tensor:
        if indices is None:
            return x
        if max(indices) >= x.shape[-1]:
            raise ValueError(f"Cannot select indices {indices} from tensor with last dim {x.shape[-1]}.")
        idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
        return x.index_select(dim=x.ndim - 1, index=idx)

    @staticmethod
    def _scatter(x: torch.Tensor, indices: list[int] | None, original_dim: int | None) -> torch.Tensor:
        if indices is None or original_dim is None:
            return x
        idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
        out = x.new_zeros(x.shape[:-1] + (int(original_dim),))
        out.index_copy_(dim=x.ndim - 1, index=idx, source=x)
        return out

    def forward(self, batch):
        if "action" in batch and self.action_indices is not None:
            batch["action"][self.action_key] = self._select(batch["action"][self.action_key], self.action_indices)
        if self.state_indices is not None:
            batch["state"][self.state_key] = self._select(batch["state"][self.state_key], self.state_indices)
        return batch

    def backward(self, batch):
        if "action" in batch and self.action_indices is not None:
            batch["action"][self.action_key] = self._scatter(
                batch["action"][self.action_key],
                self.action_indices,
                self.original_action_dim,
            )
        if "state" in batch and self.state_indices is not None:
            batch["state"][self.state_key] = self._scatter(
                batch["state"][self.state_key],
                self.state_indices,
                self.original_state_dim,
            )
        return batch


class WrapStateAngle:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch
    
    def backward(self, batch):
        return batch
