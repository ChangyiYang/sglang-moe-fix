"""Scheduler-side soft-prompt handling: attach to a Req and emit prefill overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.soft_prompt.pool import get_soft_prompt_pool
from sglang.srt.soft_prompt.utils import SOFT_PROMPT_CACHE_KEY


@dataclass(frozen=True)
class SoftPromptRef:
    name: str
    version: Optional[int]
    start: int  # absolute position of the first placeholder token
    length: int


def attach_soft_prompt(req, recv_req) -> None:
    """Record the request's soft prompt and namespace its prefix-cache key by it.

    Placeholder tokens are identical across tenants, so without the namespace two tenants
    with the same text would share KV computed from different embeddings.
    """
    pool = get_soft_prompt_pool()
    entry = pool.entries.get(recv_req.soft_prompt) if pool is not None else None
    if entry is None:
        req.set_finish_with_abort(
            f"soft prompt {recv_req.soft_prompt!r} is not loaded on this worker"
        )
        return
    req.soft_prompt = SoftPromptRef(
        name=recv_req.soft_prompt,
        version=recv_req.soft_prompt_version,
        start=recv_req.soft_prompt_start,
        length=entry.length,
    )
    namespace = SOFT_PROMPT_CACHE_KEY.format(
        name=recv_req.soft_prompt, version=recv_req.soft_prompt_version
    )
    req.extra_key = (
        namespace if req.extra_key is None else f"{req.extra_key}|{namespace}"
    )


def soft_prompt_extend_overrides(
    req, pre_len: int, extend_len: int, flat_offset: int
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """Embeddings and flat extend-batch positions for the placeholders in this chunk.

    ``pre_len`` is the cached-prefix length, ``extend_len`` the number of tokens computed
    now, and ``flat_offset`` where this request's tokens start in the flattened batch.
    """
    ref: SoftPromptRef = req.soft_prompt
    lo = max(ref.start, pre_len)
    hi = min(ref.start + ref.length, pre_len + extend_len)
    if lo >= hi:
        return None
    offsets = list(range(lo - ref.start, hi - ref.start))
    positions = [flat_offset + (p - pre_len) for p in range(lo, hi)]
    embeds = get_soft_prompt_pool().gather(ref.name, ref.version, offsets)
    return embeds, positions
