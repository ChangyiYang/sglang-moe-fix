"""CPU unit tests for the multi-tenant soft-prompt pool and prefill-override clipping."""

import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.soft_prompt import pool as pool_mod
from sglang.srt.soft_prompt.pool import SoftPromptPool, read_soft_prompt_tensor
from sglang.srt.soft_prompt.scheduling import (
    SoftPromptRef,
    soft_prompt_extend_overrides,
)
from sglang.srt.soft_prompt.utils import parse_soft_prompt_paths, soft_prompt_length

H = 8


def _pool(max_prompts=2, max_len=4):
    return SoftPromptPool(max_prompts, max_len, H, torch.float32, torch.device("cpu"))


def test_load_gather_and_slot_reuse():
    p = _pool()
    a = torch.arange(3 * H, dtype=torch.float32).view(3, H)
    b = -torch.arange(2 * H, dtype=torch.float32).view(2, H)
    p.load("a", a, version=1)
    p.load("b", b, version=2)
    assert p.loaded() == {"a": 3, "b": 2}
    assert torch.equal(p.gather("a", 1, [0, 2]), a[[0, 2]])
    assert torch.equal(p.gather("b", 2, [1]), b[[1]])
    with pytest.raises(ValueError, match="full"):
        p.load("c", a, version=3)
    p.unload("a")
    p.load("c", b, version=3)  # reuses a's slot
    assert torch.equal(p.gather("c", 3, [0, 1]), b)
    with pytest.raises(ValueError, match="unloaded"):
        p.gather("a", 1, [0])


def test_reload_in_place_and_validation():
    p = _pool()
    p.load("a", torch.zeros(2, H), version=1)
    slot = p.entries["a"].slot
    p.load("a", torch.ones(4, H), version=2)
    assert p.entries["a"].slot == slot and p.entries["a"].length == 4
    with pytest.raises(ValueError, match="hidden size"):
        p.load("x", torch.zeros(2, H + 1), version=1)
    with pytest.raises(ValueError, match="length"):
        p.load("x", torch.zeros(5, H), version=1)
    with pytest.raises(ValueError, match="out of range"):
        p.gather("a", 2, [4])


def test_extend_overrides_clip_to_chunk(monkeypatch):
    p = _pool(max_prompts=1, max_len=8)
    emb = torch.arange(6 * H, dtype=torch.float32).view(6, H)
    p.load("t", emb, version=0)
    monkeypatch.setattr(pool_mod, "_POOL", p)
    req = SimpleNamespace(soft_prompt=SoftPromptRef("t", 0, start=3, length=6))

    # Whole prompt in one chunk, no cached prefix: placeholders at absolute 3..8.
    e, pos = soft_prompt_extend_overrides(req, pre_len=0, extend_len=12, flat_offset=100)
    assert pos == list(range(103, 109)) and torch.equal(e, emb)

    # Prefix cache covers absolute 0..4 -> only offsets 2..5 are recomputed.
    e, pos = soft_prompt_extend_overrides(req, pre_len=5, extend_len=7, flat_offset=0)
    assert pos == [0, 1, 2, 3] and torch.equal(e, emb[2:6])

    # Chunked prefill: first chunk covers absolute 0..4 only.
    e, pos = soft_prompt_extend_overrides(req, pre_len=0, extend_len=5, flat_offset=7)
    assert pos == [10, 11] and torch.equal(e, emb[0:2])

    # Prompt entirely inside the cached prefix -> nothing to replace.
    assert soft_prompt_extend_overrides(req, pre_len=9, extend_len=3, flat_offset=0) is None


def test_file_helpers(tmp_path):
    f = tmp_path / "sp.safetensors"
    save_file({"soft_prompt": torch.randn(5, H)}, str(f))
    assert soft_prompt_length(str(f)) == 5
    assert read_soft_prompt_tensor(str(f)).shape == (5, H)
    assert parse_soft_prompt_paths([f"a={f}"]) == [("a", str(f))]
    with pytest.raises(ValueError):
        parse_soft_prompt_paths(["no-equals"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_soft_prompt_paths([f"a={f}", f"a={f}"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
