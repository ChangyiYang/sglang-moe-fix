"""Scheduler-side soft-prompt handling: attach to a Req and emit prefill overrides."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.soft_prompt.pool import get_soft_prompt_pool
from sglang.srt.soft_prompt.utils import is_soft_prompt_id


@dataclass(frozen=True)
class SoftPromptRef:
    name: str
    version: Optional[int]
    start: int  # absolute position of the first placeholder token
    length: int


def attach_soft_prompt(req, recv_req) -> None:
    """Record the request's soft prompt (pool reference and placeholder span).

    Prefix-cache isolation needs no extra key: the soft-prompt span already carries
    per-prompt, per-version token ids (see soft_prompt.utils.SOFT_PROMPT_ID_BASE), so the
    radix tree diverges exactly where the soft prompt starts and tokens before it stay
    shareable across tenants.
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


def map_soft_prompt_ids_for_model(token_ids, placeholder_id: int):
    """Replace per-prompt soft-prompt ids (outside the vocabulary) by the placeholder id."""
    if not any(is_soft_prompt_id(t) for t in token_ids):
        return token_ids
    mapped = [placeholder_id if is_soft_prompt_id(t) else t for t in token_ids]
    return array("q", mapped) if isinstance(token_ids, array) else mapped
