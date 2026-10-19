"""Helpers shared by the tokenizer (CPU) and model-runner (GPU) sides of soft prompts."""

from __future__ import annotations

from typing import List, Optional, Tuple

from safetensors import safe_open

# Text marker replaced by the placeholder-id run of the request's soft prompt.
SOFT_PROMPT_MARKER = "<|soft_prompt|>"
# Logit bias that keeps the input-only placeholder id from ever being sampled.
SOFT_PROMPT_LOGIT_BIAS = -1.0e4
# Soft-prompt positions are stored in a request's token ids as per-prompt ids outside the
# vocabulary (like multimodal pad values), so the prefix cache keys them per prompt and
# version while tokens before the soft prompt stay shareable across tenants. The model
# never sees these ids: they are mapped back to the placeholder id when building its input.
# Range: above multimodal pad values ([1e6, 1e6 + 2**30)) and below int32 max.
SOFT_PROMPT_ID_BASE = 1_200_000_000
SOFT_PROMPT_ID_LIMIT = 2**31 - 1


def soft_prompt_token_ids(version: int, length: int, max_len: int) -> List[int]:
    """Per-prompt token ids for one (re)load of a soft prompt."""
    start = SOFT_PROMPT_ID_BASE + version * max_len
    if start + length > SOFT_PROMPT_ID_LIMIT:
        raise ValueError(
            "soft prompt id space exhausted; restart the server to reset versions"
        )
    return list(range(start, start + length))


def is_soft_prompt_id(token_id: int) -> bool:
    return token_id >= SOFT_PROMPT_ID_BASE


def parse_soft_prompt_paths(paths: Optional[List[str]]) -> List[Tuple[str, str]]:
    parsed = []
    for item in paths or []:
        if "=" not in item:
            raise ValueError(
                f"--soft-prompt-paths entries must be NAME=PATH, got {item!r}"
            )
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(
                f"--soft-prompt-paths entries must be NAME=PATH, got {item!r}"
            )
        parsed.append((name, path))
    names = [n for n, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError("--soft-prompt-paths has duplicate names")
    return parsed


def soft_prompt_length(path: str) -> int:
    """Prompt length read from the safetensors header (no tensor data is loaded)."""
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(
                f"soft prompt file {path} must hold exactly one tensor, found {keys}"
            )
        shape = f.get_slice(keys[0]).get_shape()
    if len(shape) != 2:
        raise ValueError(
            f"soft prompt tensor in {path} must be [length, hidden_size], got {shape}"
        )
    return int(shape[0])
