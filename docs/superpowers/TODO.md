# Disaggregated Prefill — TODO / known limitations

Tracked follow-ups for the `disaggregated-prefill` branch. Each entry is
self-contained (assume the reader has no prior context) with exact refs.

---

## 1. Stream-layout-agnostic delegated KV transfer (drop the `--kv-unified` coupling)

**Status:** known limitation; workaround in place (`--kv-unified` on both peers).
Not a correctness bug — delegation silently falls back to local prefill, output
stays correct.

**Symptom:** when the prefill server and decode client have different KV cache
stream counts (`n_stream`), every delegation fails at state-load and falls back:

```
E state_seq_set_data: error loading state: n_stream mismatch   → prefill_fallback_total++
```

Most common trigger: prefill server run with `-np N` (N>1) *without*
`--kv-unified` (cache is non-unified, n_stream=N) against a `--parallel 1`
decode client (n_stream=1).

**Root cause** — `llama_kv_cache::state_read`, `src/llama-kv-cache.cpp:2063`:

```cpp
uint32_t n_stream_cur;
io.read(&n_stream_cur, sizeof(n_stream_cur));
if (n_stream_cur != n_stream) throw std::runtime_error("n_stream mismatch");
```

`n_stream` is a cache *partitioning* (how the buffer splits into parallel
sequence streams), **not a capacity**. A single sequence's K/V vectors are
identical regardless of `n_stream`, so this equality guard is conservative: the
state format was built for same-process `/slots` save/restore, where `n_stream`
always matches, and never had to translate layouts. It is NOT a fundamental
incompatibility.

**Note on `n_ctx` — already adaptive, do NOT add it as a gate.** `dp_hello_compatible`
(`tools/server/server-prefill.cpp:550`) does *not* compare `n_ctx`; it only
checks proto_version / arch / n_layer / n_embd / n_head_kv / type_k / type_v
(+ lora_hash, warning only). `n_ctx` is advertised in the descriptor purely as
information (how big a prompt the server can prefill). It is a **capacity**, so
it adapts on its own: the transferred blob is per-sequence (sized by the prompt's
tokens), and any cache with enough free cells holds it. The only real constraint
is per-sequence fit, enforced independently on each side via `EXCEED_CONTEXT`
(prompt must fit the server's slot to be prefilled AND the client's slot to be
held → effective max = `min(server n_ctx_slot, client n_ctx_slot)`). Both
directions (server larger or client larger) already work. `n_stream` is a layout,
which is why it doesn't adapt; `n_ctx` is capacity, which is why it does.

**Two gaps:**

1. **Not caught at handshake.** The compat descriptor (`/v1/models` → `prefill`)
   advertises `n_stream`-relevant nothing, so `dp_hello_compatible` passes and the
   failure only surfaces mid-stream as a fallback — invisible unless you watch
   `prefill_fallback_total`.
2. **No cross-layout remap**, so peers are forced to match `n_stream` via
   `--kv-unified` on both sides — undocumented coupling.

**Fix (two tiers):**

- **Tier 1 — fail fast (small):** add `n_stream` (or `kv_unified`) to the compat
  descriptor and `dp_hello_compatible`. Mismatch → `400` at delegation start with
  a clear message ("prefill n_stream=4 ≠ decode n_stream=1; run both with
  `--kv-unified`") instead of a silent fallback. Document the requirement in
  `docs/disaggregated-prefill.md`.
- **Tier 2 — remove the coupling (proper):** make **single-sequence** `state_read`
  stream-layout-agnostic — remap incoming cells into the local cache's stream
  structure regardless of source `n_stream` (per-cell K/V payload is unchanged;
  only `seq_to_stream` placement differs). Gate it to the single-`seq_id` path so
  whole-cache save/restore keeps the strict check.

**Acceptance:** owlai `-np 4` *without* `--kv-unified` → `--parallel 1` decode
client delegation round-trips, coherent output, `prefill_fallback_total == 0`
(Tier 2); or a clean `400` naming the mismatch (Tier 1).

**Refs:** `src/llama-kv-cache.cpp:2063` (check); `tools/server/server-prefill.cpp:550`
(`dp_hello_compatible` + descriptor); `tools/server/server-context.cpp` (compat
check on delegate). Verified live 2026-07-13: `--kv-unified` on both peers is the
current workaround.

---

## 2. `--parallel-decode` / single-flight decode — finish + validate at scale

**Status:** implemented on-branch, **uncommitted**, mechanism verified on a
micro-test; not yet benchmarked under sustained load.

**What it is:** a new server flag `-npd / --parallel-decode N` (default `-1` =
`--parallel`) that caps how many slots may be *generating* concurrently, while
leaving prefill/prefetch parallelism at `--parallel`. Purpose: with speculative
(MTP) decode, two concurrent decode streams on one GPU collide and per-request
t/s collapses (measured 47→17 under the noisy c6 benchy; ~26 each in a clean
2-on-1 test). `--parallel-decode 1` keeps one clean MTP stream per GPU while the
other slot prefetches its KV on the remote prefill server — hiding prefill
latency without the collision.

**Implementation:** `common/common.h` (`n_parallel_decode`), `common/arg.cpp`
(the flag), and the gate in `tools/server/server-context.cpp` (in the
`iterate(slots, …)` loop before the STARTED prompt-processing block): a delegated
slot whose remote prefill finished (`dp_finished_id_task == task->id`, KV loaded)
is held with `return` while `n_generating >= limit`. `is_processing()` stays true
for the held slot, so the main loop keeps running and re-admits it when a decode
slot frees.

**Verified (2026-07-13, micro-test):** 2 concurrent long reqs to one endpoint,
both delegated, decode serialized → 51.9 / 47.6 t/s per req (vs ~26 concurrent),
staggered completion (serial). Deployed config on the 3 SYCL decoders:
`--parallel 2 --kv-unified --parallel-decode 1 --prefill-rpc-max-inflight 2`.

**Open items:**

- **Scope:** the gate only fires for delegated (`dp_resume`) slots. A slot that
  prefills *locally* is not held and will decode concurrently. Fine for the
  all-delegated topology; document the limitation (or extend the gate to the
  DONE_PROMPT→GENERATING transition for the general case, handling the ephemeral
  logits carefully — harder, see the resume-path rationale).
- **Prerequisite:** useful only with `--prefill-rpc-max-inflight ≥ --parallel`
  (default is **1**, `common/common.h:678`), else the 2nd slot can't delegate and
  prefills locally. Consider defaulting `max-inflight` to `n_parallel` when
  delegation is on, or warn when `npd>0 && max-inflight < n_parallel`.
- **Benchmark at scale (the real test):** run benchy across all 3 endpoints,
  `--parallel-decode 1` vs `2`, compare total t/s + TTFR + owlai occupancy
  overlap. Aggregate throughput is genuinely uncertain — a single 50 t/s stream
  vs two ~26 t/s streams is close on one GPU; single-flight's win may be
  per-request latency + keeping the prefill server fed, not raw aggregate.
- **Tests:** add a server test asserting `n_generating` never exceeds
  `--parallel-decode` and that delegation still round-trips.

**Refs:** `common/common.h` (`n_parallel_decode`, `prefill_rpc_max_inflight:678`),
`common/arg.cpp` (`--parallel-decode`), `tools/server/server-context.cpp` (gate),
`server-prefill.cpp:97` (inflight limit).
