"""End-to-end check of multi-tenant soft prompts against a virtual-token reference.

Reference: the same soft prompt baked into a model checkpoint as extra vocabulary rows
("virtual tokens"), served as ordinary token ids. Candidate: the base model with the prompt
in the soft-prompt pool. Greedy outputs must match token for token.

Env:
  SP_BASE_MODEL   base model path (e.g. Qwen3-8B)
  SP_VIRTUAL_MODEL exported reference model dir containing soft_prompt_tokens.json
  SP_TENSOR       safetensors with the [length, hidden] soft prompt
  SP_PLACEHOLDER  placeholder token id for the pool path (an unused padded-vocab id)
  SP_EAGLE3_DRAFT / SP_DFLASH_DRAFT  optional drafter paths for the speculative test
Run on a GPU node:  python test_soft_prompt_e2e.py
"""

import json
import os
import subprocess
import sys
import time
import unittest

import requests
from safetensors.torch import load_file, save_file

PORT = int(os.environ.get("SP_PORT", "31555"))
URL = f"http://127.0.0.1:{PORT}"
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain what a prefix cache is in one paragraph.",
    "Given an array of integers, find two numbers that add up to a target.",
]


def _launch(model, extra):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--port",
        str(PORT),
        "--mem-fraction-static",
        "0.6",
        "--disable-cuda-graph",
        "--random-seed",
        "0",
        *extra,
    ]
    proc = subprocess.Popen(cmd)
    for _ in range(600):
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()


def _generate(input_ids, soft_prompt=None, max_new=96, logit_bias=None, logprob=True):
    sp = {"temperature": 0, "max_new_tokens": max_new}
    if logit_bias:
        sp["logit_bias"] = logit_bias
    body = {"input_ids": input_ids, "sampling_params": sp, "return_logprob": logprob}
    if soft_prompt:
        body["soft_prompt"] = soft_prompt
    r = requests.post(f"{URL}/generate", json=body, timeout=600)
    r.raise_for_status()
    out = r.json()
    return out["output_ids"], out["meta_info"]


class TestSoftPromptE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.base = os.environ["SP_BASE_MODEL"]
        cls.virtual = os.environ["SP_VIRTUAL_MODEL"]
        cls.placeholder = int(os.environ["SP_PLACEHOLDER"])
        cls.meta = json.load(open(os.path.join(cls.virtual, "soft_prompt_tokens.json")))
        cls.tensor_path = os.environ["SP_TENSOR"]
        cls.tok = AutoTokenizer.from_pretrained(cls.base)

    def _prompt_ids(self, text, run, system=None):
        head = "<|im_start|>user\n"
        if system:
            head = f"<|im_start|>system\n{system}<|im_end|>\n" + head
        pre = self.tok.encode(head, add_special_tokens=False)
        post = self.tok.encode(
            text + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )
        return pre + run + post

    def test_matches_virtual_token_reference(self):
        vids = self.meta["token_ids"]
        length = len(vids)
        # Reference outputs from the virtual-token model.
        proc = _launch(self.virtual, [])
        try:
            bias = {str(t): -1e4 for t in vids}
            ref = [
                _generate(self._prompt_ids(p, vids), logit_bias=bias)[0]
                for p in PROMPTS
            ]
        finally:
            _stop(proc)

        # Same prompts through the pool on the base model; a second tenant gets a
        # perturbed prompt to check cache isolation.
        tensor = next(iter(load_file(self.tensor_path).values()))
        other = os.path.join(
            os.path.dirname(self.tensor_path), "_sp_e2e_other.safetensors"
        )
        save_file({"soft_prompt": tensor.flip(0).contiguous()}, other)
        proc = _launch(
            self.base,
            [
                "--enable-soft-prompt",
                "--soft-prompt-placeholder-id",
                str(self.placeholder),
                "--max-soft-prompt-len",
                str(max(length, 256)),
                "--soft-prompt-paths",
                f"tenant_a={self.tensor_path}",
            ],
        )
        try:
            r = requests.post(
                f"{URL}/load_soft_prompt",
                json={"name": "tenant_b", "path": other},
                timeout=120,
            )
            self.assertEqual(r.status_code, 200, r.text)
            run = [self.placeholder] * length
            got, metas = [], []
            for p in PROMPTS:
                out, meta = _generate(self._prompt_ids(p, run), soft_prompt="tenant_a")
                got.append(out)
                metas.append(meta)
            for i, (a, b) in enumerate(zip(ref, got)):
                self.assertEqual(
                    a,
                    b,
                    f"prompt {i}: pool output differs from virtual-token reference",
                )

            # Same tenant again: the soft-prompt prefix must come from the cache.
            _, meta = _generate(
                self._prompt_ids(PROMPTS[0], run), soft_prompt="tenant_a"
            )
            self.assertGreaterEqual(meta.get("cached_tokens", 0), length)
            # Other tenant, identical ids: must not reuse tenant_a's KV and must differ.
            out_b, meta_b = _generate(
                self._prompt_ids(PROMPTS[0], run), soft_prompt="tenant_b"
            )
            self.assertLess(meta_b.get("cached_tokens", 0), length)
            self.assertNotEqual(out_b, got[0])
            # A shared system prompt before the soft prompt is cached across tenants,
            # while each tenant's soft-prompt span stays private.
            system = " ".join(
                ["You are a careful, concise programming assistant."] * 40
            )
            sys_len = len(
                self.tok.encode(
                    f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
                    add_special_tokens=False,
                )
            )
            _generate(self._prompt_ids(PROMPTS[2], run, system), soft_prompt="tenant_a")
            _, meta_shared = _generate(
                self._prompt_ids(PROMPTS[2], run, system), soft_prompt="tenant_b"
            )
            cached = meta_shared.get("cached_tokens", 0)
            # Prefix matches are page-aligned (e.g. page_size 64 with trtllm_mha).
            page = int(
                requests.get(f"{URL}/get_server_info", timeout=30)
                .json()
                .get("page_size")
                or 1
            )
            self.assertGreaterEqual(cached, (sys_len - 1) // page * page)
            self.assertGreater(cached, 0)
            self.assertLess(cached, sys_len + length)
            # Reloading a prompt changes its ids: its old cached span is not reused.
            r = requests.post(
                f"{URL}/load_soft_prompt",
                json={"name": "tenant_a", "path": self.tensor_path},
                timeout=120,
            )
            self.assertEqual(r.status_code, 200, r.text)
            out_re, meta_re = _generate(
                self._prompt_ids(PROMPTS[0], run), soft_prompt="tenant_a"
            )
            self.assertLess(meta_re.get("cached_tokens", 0), length)
            self.assertEqual(out_re, got[0])  # same tensor -> same output
            # Text marker path expands to the same ids.
            text = (
                "<|im_start|>user\n<|soft_prompt|>"
                + PROMPTS[0]
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            r = requests.post(
                f"{URL}/generate",
                json={
                    "text": text,
                    "soft_prompt": "tenant_a",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 96},
                },
                timeout=600,
            )
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["output_ids"], got[0])
            # Batched request: one soft-prompt item and one plain item.
            plain = self._prompt_ids(PROMPTS[1], [])
            r = requests.post(
                f"{URL}/generate",
                json={
                    "input_ids": [self._prompt_ids(PROMPTS[0], run), plain],
                    "soft_prompt": ["tenant_a", None],
                    "sampling_params": {"temperature": 0, "max_new_tokens": 96},
                },
                timeout=600,
            )
            self.assertEqual(r.status_code, 200, r.text)
            batch = r.json()
            self.assertEqual(batch[0]["output_ids"], got[0])
            ref_plain, _ = _generate(plain)
            self.assertEqual(batch[1]["output_ids"], ref_plain)
            # Unloaded / unknown prompt is rejected cleanly.
            r = requests.post(
                f"{URL}/unload_soft_prompt", json={"name": "tenant_b"}, timeout=60
            )
            self.assertEqual(r.status_code, 200, r.text)
            r = requests.post(
                f"{URL}/generate",
                json={
                    "input_ids": self._prompt_ids(PROMPTS[0], run),
                    "soft_prompt": "tenant_b",
                    "sampling_params": {"max_new_tokens": 4},
                },
                timeout=60,
            )
            self.assertNotEqual(r.status_code, 200)
        finally:
            _stop(proc)
            os.remove(other)

    def test_spec_decoding_matches_virtual_reference(self):
        """Speculative decoding on soft-prompt requests is lossless and drafts from the
        same inputs as the virtual-token reference (identical verify-step counts)."""
        drafters = []
        if os.environ.get("SP_EAGLE3_DRAFT"):
            drafters.append(
                (
                    "eagle3",
                    True,
                    [
                        "--speculative-algorithm",
                        "EAGLE3",
                        "--speculative-draft-model-path",
                        os.environ["SP_EAGLE3_DRAFT"],
                        "--speculative-num-steps",
                        "3",
                        "--speculative-eagle-topk",
                        "1",
                        "--speculative-num-draft-tokens",
                        "4",
                    ],
                )
            )
        if os.environ.get("SP_DFLASH_DRAFT"):
            drafters.append(
                (
                    "dflash",
                    False,
                    [
                        "--speculative-algorithm",
                        "DFLASH",
                        "--speculative-draft-model-path",
                        os.environ["SP_DFLASH_DRAFT"],
                    ],
                )
            )
        if not drafters:
            self.skipTest("set SP_EAGLE3_DRAFT and/or SP_DFLASH_DRAFT")
        vids = self.meta["token_ids"]
        length = len(vids)
        run = [self.placeholder] * length
        bias = {str(t): -1e4 for t in vids}
        for name, exact_drafting, spec_args in drafters:
            proc = _launch(self.virtual, spec_args)
            try:
                ref = [
                    _generate(
                        self._prompt_ids(p, vids),
                        logit_bias=bias,
                        max_new=192,
                        logprob=False,
                    )
                    for p in PROMPTS
                ]
            finally:
                _stop(proc)
            proc = _launch(
                self.base,
                spec_args
                + [
                    "--enable-soft-prompt",
                    "--soft-prompt-placeholder-id",
                    str(self.placeholder),
                    "--max-soft-prompt-len",
                    str(max(length, 256)),
                    "--soft-prompt-paths",
                    f"tenant_a={self.tensor_path}",
                ],
            )
            try:
                got = [
                    _generate(
                        self._prompt_ids(p, run),
                        soft_prompt="tenant_a",
                        max_new=192,
                        logprob=False,
                    )
                    for p in PROMPTS
                ]
            finally:
                _stop(proc)
            for i, ((ref_ids, ref_meta), (ids, meta)) in enumerate(zip(ref, got)):
                print(
                    f"[{name}] prompt {i}: verify_ct ref={ref_meta.get('spec_verify_ct')} "
                    f"pool={meta.get('spec_verify_ct')} "
                    f"accept ref={ref_meta.get('spec_accept_length')} "
                    f"pool={meta.get('spec_accept_length')}",
                    flush=True,
                )
                self.assertEqual(ids, ref_ids, f"[{name}] prompt {i}: outputs differ")
                if exact_drafting:
                    self.assertEqual(
                        meta.get("spec_verify_ct"),
                        ref_meta.get("spec_verify_ct"),
                        f"[{name}] prompt {i}: drafter saw different inputs",
                    )


if __name__ == "__main__":
    unittest.main()
