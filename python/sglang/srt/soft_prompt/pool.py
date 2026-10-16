"""GPU-resident pool of multi-tenant soft prompts.

A soft prompt is a learned ``[length, hidden_size]`` tensor that replaces the input
embeddings of a contiguous run of placeholder tokens. Prompts are input-only: they never
touch the LM head or the vocabulary, so each tenant costs ``length * hidden_size`` elements.

The pool is a flat ``[max_prompts * max_len, hidden_size]`` buffer; slot ``s`` owns rows
``[s * max_len, s * max_len + length)``. Every (re)load bumps a version that is folded into
the request's prefix-cache key, so cached KV computed from an older tensor is never reused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


@dataclass
class SoftPromptEntry:
    slot: int
    length: int
    version: int


def read_soft_prompt_tensor(path: str) -> torch.Tensor:
    tensors = load_file(path, device="cpu")
    if len(tensors) != 1:
        raise ValueError(
            f"soft prompt file {path} must hold exactly one tensor, found {sorted(tensors)}"
        )
    tensor = next(iter(tensors.values()))
    if tensor.ndim != 2:
        raise ValueError(
            f"soft prompt tensor in {path} must be [length, hidden_size], got {tuple(tensor.shape)}"
        )
    return tensor


class SoftPromptPool:
    def __init__(
        self,
        max_prompts: int,
        max_len: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if max_prompts <= 0 or max_len <= 0:
            raise ValueError("max_soft_prompts and max_soft_prompt_len must be positive")
        self.max_prompts = max_prompts
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.buffer = torch.zeros(
            max_prompts * max_len, hidden_size, dtype=dtype, device=device
        )
        self.entries: Dict[str, SoftPromptEntry] = {}
        self.free_slots: List[int] = list(range(max_prompts))
        logger.info(
            "Soft prompt pool: %d slots x %d tokens x %d hidden (%.1f MiB, %s)",
            max_prompts,
            max_len,
            hidden_size,
            self.buffer.numel() * self.buffer.element_size() / 2**20,
            dtype,
        )

    def load(self, name: str, tensor: torch.Tensor, version: int) -> SoftPromptEntry:
        length, hidden = tensor.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"soft prompt {name!r} hidden size {hidden} != model hidden size {self.hidden_size}"
            )
        if not 0 < length <= self.max_len:
            raise ValueError(
                f"soft prompt {name!r} length {length} must be in [1, {self.max_len}]"
            )
        entry = self.entries.get(name)
        if entry is not None:
            slot = entry.slot  # reload in place
        elif self.free_slots:
            slot = self.free_slots.pop(0)
        else:
            raise ValueError(
                f"soft prompt pool is full ({self.max_prompts} prompts); unload one first"
            )
        start = slot * self.max_len
        # Same cast as exporting the prompt into an embedding table of the model dtype.
        self.buffer[start : start + length].copy_(
            tensor.to(device=self.device, dtype=self.dtype)
        )
        entry = SoftPromptEntry(slot=slot, length=length, version=version)
        self.entries[name] = entry
        return entry

    def unload(self, name: str) -> None:
        entry = self.entries.pop(name, None)
        if entry is None:
            raise ValueError(f"soft prompt {name!r} is not loaded")
        self.free_slots.append(entry.slot)

    def loaded(self) -> Dict[str, int]:
        return {name: e.length for name, e in self.entries.items()}

    def gather(
        self, name: str, version: Optional[int], offsets: Sequence[int]
    ) -> torch.Tensor:
        """Embedding rows for prompt-relative offsets, shape [len(offsets), hidden]."""
        entry = self.entries.get(name)
        if entry is None:
            raise ValueError(
                f"soft prompt {name!r} was unloaded while a request using it was in flight"
            )
        if version is not None and entry.version != version:
            # The prompt was reloaded after tokenization. The request's cache key carries the
            # old version, so nothing is shared incorrectly; it simply runs on the new tensor.
            logger.warning(
                "soft prompt %r reloaded (v%s -> v%s) while a request was pending",
                name,
                version,
                entry.version,
            )
        if any(not 0 <= o < entry.length for o in offsets):
            raise ValueError(f"soft prompt {name!r} offset out of range")
        rows = torch.tensor(
            [entry.slot * self.max_len + o for o in offsets],
            dtype=torch.long,
            device=self.device,
        )
        return self.buffer.index_select(0, rows)


_POOL: Optional[SoftPromptPool] = None


def set_soft_prompt_pool(pool: Optional[SoftPromptPool]) -> None:
    global _POOL
    _POOL = pool


def get_soft_prompt_pool() -> Optional[SoftPromptPool]:
    return _POOL
