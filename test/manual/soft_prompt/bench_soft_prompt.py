"""Quick throughput check: same server, requests with vs without a pooled soft prompt.

Assumes a server launched with --enable-soft-prompt and tenants t0..t{N-1} loaded.
Usage: python bench_soft_prompt.py --model <tok path> --placeholder ID --sp-len L --tenants N
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=31555)
ap.add_argument("--model", required=True)
ap.add_argument("--placeholder", type=int, required=True)
ap.add_argument("--sp-len", type=int, default=256)
ap.add_argument("--tenants", type=int, default=8)
ap.add_argument("--num", type=int, default=256)
ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--in-len", type=int, default=512)
ap.add_argument("--out-len", type=int, default=256)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
rng = random.Random(0)


def ids(with_sp):
    head = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    body = [rng.randrange(1000, 100000) for _ in range(a.in_len)]
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    return head + ([a.placeholder] * a.sp_len if with_sp else []) + body + tail


async def run(with_sp):
    reqs = [(ids(with_sp), f"t{i % a.tenants}" if with_sp else None) for i in range(a.num)]
    sem = asyncio.Semaphore(a.conc)
    ttft = []

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async def one(input_ids, sp):
            async with sem:
                body = {"input_ids": input_ids, "stream": True,
                        "sampling_params": {"temperature": 0, "max_new_tokens": a.out_len,
                                            "ignore_eos": True}}
                if sp:
                    body["soft_prompt"] = sp
                t0 = time.perf_counter()
                first = None
                async with s.post(f"http://127.0.0.1:{a.port}/generate", json=body) as r:
                    r.raise_for_status()
                    async for line in r.content:
                        if first is None and line.startswith(b"data:"):
                            first = time.perf_counter() - t0
                ttft.append(first)

        t0 = time.perf_counter()
        await asyncio.gather(*[one(i, sp) for i, sp in reqs])
        dur = time.perf_counter() - t0
    ttft.sort()
    return {"soft_prompt": with_sp, "out_tok_per_s": round(a.num * a.out_len / dur),
            "ttft_p50_ms": round(ttft[len(ttft) // 2] * 1000, 1), "duration_s": round(dur, 1)}


for with_sp in (False, True, False, True):
    print(json.dumps(asyncio.run(run(with_sp))), flush=True)
