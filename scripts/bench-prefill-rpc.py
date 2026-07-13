#!/usr/bin/env python3
"""TTFT benchmark for disaggregated prefill.

Usage:
  # terminal 1: prefill server on the fast box (serves POST /v1/prefill on its main port)
  #   llama-server -m model.gguf --host 0.0.0.0 --port 8080 --prefill-serve
  # terminal 2: decode server WITHOUT --prefill-rpc                 -> baseline (port 8090)
  # terminal 3: decode server WITH --prefill-rpc gpubox:8080        -> delegated (port 8091)
  ./scripts/bench-prefill-rpc.py --url http://localhost:8090 --label baseline
  ./scripts/bench-prefill-rpc.py --url http://localhost:8091 --label delegated
"""
import argparse
import json
import time
import urllib.request

WORD = "the quick brown fox jumps over the lazy dog "


def run(url: str, n_words: int) -> dict:
    prompt = WORD * n_words
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": False,  # force a full prefill every run
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    wall_ms = (time.time() - t0) * 1000
    t = res.get("timings", {})
    return {
        "n_prompt": t.get("prompt_n", -1),
        "prompt_ms": t.get("prompt_ms", -1.0),
        "wall_ms": wall_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--sizes", default="64,256,1024,4096,16384")
    args = ap.parse_args()

    print(f"{'label':<12} {'n_prompt':>9} {'prompt_ms':>12} {'wall_ms':>10}")
    for n_words in (int(s) for s in args.sizes.split(",")):
        r = run(args.url, n_words)
        print(f"{args.label:<12} {r['n_prompt']:>9} {r['prompt_ms']:>12.1f} {r['wall_ms']:>10.1f}")


if __name__ == "__main__":
    main()
