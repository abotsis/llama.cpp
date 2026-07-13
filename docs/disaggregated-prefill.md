# Disaggregated Prefill

> [!WARNING]
> Experimental, fork-only feature. The prefill data plane rides `llama-server`'s normal
> HTTP stack, so `--api-key` (and TLS, if the server is built and run with it) protect
> `POST /v1/prefill` like any other endpoint. But the endpoint lets an authenticated
> caller compute and download the raw KV state for arbitrary token sequences, which is
> sensitive - it stays opt-in via `--prefill-serve`, off by default.

Disaggregated prefill lets one `llama-server` instance (the **decode server**, e.g. a Mac
serving users) delegate the prompt-processing phase of long requests to another
`llama-server` instance running the same model on a faster GPU (the **prefill server**).
Generation then continues locally as if the decode server had prefilled the prompt
itself.

The data plane is HTTP: the decode server issues `POST /v1/prefill` to the prefill
server's **main HTTP port** (the same port that serves completions), and the prefill
server streams the computed KV state back as the response body. There is no separate
listener, port, or protocol - the request rides the existing auth/TLS/proxy/timeout
infrastructure.

There are three KV transfer modes, chosen automatically per model:

| Mode | Model class | Transfer shape | Memory / overlap |
|------|-------------|----------------|------------------|
| **stream** | dense | attention KV as position-range chunks, streamed during prefill | O(one chunk) host memory; transfer overlaps remote compute |
| **hybrid_stream** | plain (non-SWA) hybrid, e.g. Qwen3.5/3.6, Qwen3-Next | attention KV as position-range chunks + the small recurrent state as one whole blob at the end | O(one chunk + the O(1) recurrent blob); attention transfer overlaps compute, no giant blob so no OOM at long context |
| **whole** | pure recurrent (Mamba), or any supported model when forced via `--prefill-rpc-mode whole` | the complete sequence state (plus the MTP draft-context state, if present) as one blob after prefill finishes | O(full state) host memory; no transfer/compute overlap |

Plain hybrids used to fall back to whole mode; they now stream their (rangeable, full)
attention KV like dense models and send only the small, O(1)-in-tokens recurrent state
whole. Pure recurrent models stay on whole mode. SWA / hybrid-iswa models (SWA-masked
attention is not rangeable) are excluded from delegation entirely - see Limitations. See
"How it works" for the per-mode framing.

Any failure at any point falls back to local prefill with no user-visible error.

## Quick start

Both machines must run a build of this branch, load the **same model file**, and use the
same KV cache types (`--cache-type-k/v`).

```sh
# fast GPU box (prefill server) - serves POST /v1/prefill on its main port (8080)
llama-server -m model.gguf --host 0.0.0.0 --port 8080 --prefill-serve

# user-facing box (decode server) - points --prefill-rpc at the peer's main port
llama-server -m model.gguf --prefill-rpc gpubox:8080 --prefill-rpc-min-tokens 512 --metrics
```

Send a long prompt to the decode server. Its log shows:

```
delegating prefill, n_tokens = 4096, p0 = 0
remote prefill complete: 4096 tokens, 512.00 MiB received, 1234.5 ms
```

and the prefill server logs the served request on its side.

## Flags

| Flag | Side | Env var | Default | Meaning |
|------|------|---------|---------|---------|
| `--prefill-serve` | prefill | `LLAMA_ARG_PREFILL_SERVE` | off | boolean flag; register `POST /v1/prefill` on the main HTTP port to serve remote prefill requests |
| `--prefill-rpc HOST:PORT` | decode | `LLAMA_ARG_PREFILL_RPC` | off | delegate eligible prefills to this peer's main HTTP address |
| `--prefill-rpc-min-tokens N` | decode | `LLAMA_ARG_PREFILL_RPC_MIN_TOKENS` | 512 | minimum uncached suffix length to delegate |
| `--prefill-rpc-max-inflight N` | decode | `LLAMA_ARG_PREFILL_RPC_MAX_INFLIGHT` | 1 | max concurrent delegations |
| `--prefill-rpc-mode auto\|stream\|whole\|hybrid_stream` | decode | `LLAMA_ARG_PREFILL_RPC_MODE` | auto | force the transfer mode instead of deriving it from the model; `whole` on any non-SWA model is valid (useful for testing/benchmarking mode overhead), `stream` on a hybrid/recurrent model is rejected at startup, and `hybrid_stream` is honored only if the peer's model derives it too (a plain non-SWA hybrid) |
| `--prefill-rpc-api-key KEY` | decode | `LLAMA_ARG_PREFILL_RPC_API_KEY` | none | sent as `Authorization: Bearer KEY` on `POST /v1/prefill` |

A server may both serve and delegate (`--prefill-serve` plus `--prefill-rpc`).

> [!NOTE]
> Upgrading from a pre-HTTP build: `--prefill-serve` is now a boolean flag, not an
> address. An old preset value like `--prefill-serve HOST:PORT` parses the `HOST:PORT`
> as a stray positional argument (ignored or an error) while the flag itself reads as
> truthy (serving on); drop the address.

## When does a request get delegated?

All of the following must hold, otherwise the request prefills locally as usual:

- the request is a regular completion (not embedding, rerank, or infill)
- the prompt has no multimodal chunks and no alora invocation
- the decode server has no separate draft model (`--model-draft` disables delegation at
  startup; an MTP context created via `--spec-type` is fine in whole mode - see below)
- the model is not SWA (checked once at startup; a warning is logged and delegation is
  disabled)
- the length threshold is met, and its meaning depends on the transfer mode:
  - **stream mode**: the *uncached suffix* is longer than `--prefill-rpc-min-tokens` -
    local prompt-cache prefix reuse happens first, and only what would actually need
    prefilling counts
  - **whole and hybrid_stream mode**: the *full prompt* is longer than
    `--prefill-rpc-min-tokens` - the whole sequence state (whole mode) or a fresh
    attention-KV stream from position 0 plus the recurrent tail (hybrid_stream) is
    transferred regardless of any locally cached prefix, because the recurrent state
    cannot be rewound to a cached prefix, so prefix reuse does not reduce what counts
    against the threshold
- a connection slot is free (`--prefill-rpc-max-inflight`) and the peer is not in the
  failure cooldown (see below)

The full token list is sent to the peer (so the peer's own prompt cache can reuse
prefixes across turns - chat turn N+1 delegations become incremental). In stream mode
only the range the decode server is missing is streamed back; in whole mode the peer
sends back the complete state regardless; in hybrid_stream mode the peer streams the
attention KV from position 0 as chunks and then sends the recurrent state whole.

## Requirements on the two servers

The decode server sends a compatibility descriptor in the `POST /v1/prefill` request
body; the prefill server validates it and rejects an incompatible request with a
pre-stream `400` (the decode side then falls back to local prefill):

- same protocol version (`proto`, currently 3) - **both boxes must be rebuilt from this
  branch together**, there is no forward/backward compatibility
- same model architecture, `n_layer`, `n_embd`, `n_head_kv`
- same KV cache types (`--cache-type-k`, `--cache-type-v`)
- if the decode side has an MTP draft context (`--spec-type` on the same model) and
  requests its state (`want_dft`), the prefill server must also have one; the reverse is
  fine (a serve-side MTP context that is never requested is simply not sent)

The descriptor a decode server needs in order to build a compatible request is
advertised in the `prefill` object of each item in `GET /v1/models` on the serving
side (arch, `n_layer`, `n_embd`, `n_head_kv`, `n_ctx`, `type_k`, `type_v`, `mode`,
`has_dft`, and `enabled`).

Not negotiated, must be kept in sync by the operator:

- the same model weights (the serialized KV is only meaningful for identical weights)
- LoRA configuration (a config hash mismatch logs a warning but does not block)
- the prefill server's per-slot context must fit the full prompt
  (`n_ctx / n_parallel >= prompt tokens + 1`), otherwise it replies with an error and
  the decode server falls back

The transfer is backend-portable: a CUDA box can prefill for a Metal box and vice versa.
Per-layer tensor type and row-size checks in the state restore path are the final
backstop against any undetected mismatch.

## Failure handling

Every failure ends in silent local fallback, never a user-visible error:

| Failure | Bound | Result |
|---------|-------|--------|
| peer down / connect refused | 5 s connect timeout | fallback + cooldown |
| pre-stream HTTP error (`400` compat/mode mismatch, `401` bad api key, `404` route absent, `501` serving inactive, `503` model loading) | immediate | fallback + cooldown |
| stream stalls mid-transfer (stream mode); no `PROGRESS`/`STATE` activity (whole mode) | `LLAMA_PREFILL_READ_TIMEOUT_MS` (default 120 s, per-recv) | fallback + cooldown |
| peer dies mid-stream | immediate (socket error) | fallback + cooldown |
| in-band `ERROR` frame from the peer | immediate | fallback + cooldown |
| corrupt / out-of-order / out-of-bounds chunk (stream / hybrid_stream mode) | immediate | fallback + cooldown |
| incomplete state blob (short read, whole / hybrid_stream mode) | immediate | fallback + cooldown |
| state exceeds the 2 GiB per-blob frame cap (whole mode) | immediate | serve-side error -> fallback + cooldown |
| local apply of a received whole/recurrent state fails (e.g. `set_data_ext` rejects it) | immediate | fallback, **no cooldown** - known gap, see Limitations |
| request canceled by user | immediate | no cooldown |

The read timeout is applied per receive, not to the whole transfer: as long as frames
(`CHUNK` in stream mode) or `PROGRESS` records (whole mode) keep arriving it never
fires, so it only trips on genuine mid-stream silence. There is no separate connect-then-
silence handshake timeout as in the old raw-TCP protocol - a peer that accepts the TCP
connection but never sends a byte costs the full read timeout. The connect timeout is
5 s and the request write timeout is 30 s.

- In stream mode, chunks already applied before a failure are kept: they are valid prefix
  KV, so the local fallback prefill resumes from wherever the transfer stopped - partial
  transfers are never wasted. Whole mode has no partial-progress property: fallback is
  all-or-nothing since the whole-sequence blob only applies after it arrives complete.
  Hybrid_stream is also all-or-nothing on failure: because the recurrent tail is required
  to make the attention chunks usable, a failure before the recurrent `STATE` lands drops
  the partial attention KV and rebuilds locally from scratch.
- After a failed delegation the peer is not retried for **60 seconds** (cooldown);
  requests in that window prefill locally without connection attempts. A successful
  delegation clears the cooldown. The one exception is a whole-mode (or hybrid_stream
  recurrent-tail) apply failure (the peer already reported success and closed the
  connection by the time the decode side discovers its local
  `llama_state_seq_set_data_ext` call failed): there is
  currently no cooldown on this path, so a persistently corrupt state would be
  re-requested every time. Not expected to occur in practice (the same serialize/parse
  path is exercised by the prompt cache), but flagged as a known limitation.

## Observability

With `--metrics`, the decode server exports Prometheus counters (all monotonic):

| Metric | Meaning |
|--------|---------|
| `llamacpp:prefill_delegated_total` | delegations attempted |
| `llamacpp:prefill_fallback_total` | delegations that failed and fell back |
| `llamacpp:prefill_delegated_tokens_total` | tokens whose KV was applied from a peer |
| `llamacpp:prefill_rpc_bytes_total` | KV state bytes received |

Per-delegation INFO log lines on both sides carry tokens, MiB, and elapsed ms. In
hybrid_stream mode the decode-side completion line additionally splits the received bytes
into the streamed attention KV and the whole recurrent tail, e.g. `remote prefill
complete: N tokens (hybrid-stream), X.XX MiB received (A.AA MiB attention + R.RR MiB
recurrent, dft: D bytes), T ms`, so an operator can confirm the attention half is being
streamed rather than sent whole. The `mode` is also reported per model in the `prefill`
object of `GET /v1/models` (`"stream"`, `"whole"`, or `"hybrid_stream"`).

## Benchmarking

`scripts/bench-prefill-rpc.py` measures TTFT (prompt processing time) versus prompt
length. Run the decode server once without `--prefill-rpc` (baseline) and once with it:

```sh
./scripts/bench-prefill-rpc.py --url http://localhost:8080 --label baseline
./scripts/bench-prefill-rpc.py --url http://localhost:8091 --label delegated
```

Use the resulting table to tune `--prefill-rpc-min-tokens` for your link speed: KV
transfer volume is roughly `n_layer x (n_embd_k_gqa + n_embd_v_gqa) x bytes_per_element`
per token (an 8B model with f16 KV is ~128 KB/token, so a 32k prompt moves ~4 GB - the
chunked streaming hides most of that behind remote compute, but short prompts are
cheaper to prefill locally).

## How it works

Each side derives its transfer mode from its own model at startup (`--prefill-rpc-mode`
can override the decode side): pure recurrent (Mamba) and SWA / hybrid-iswa models select
**whole** mode, plain (non-SWA) hybrids (e.g. Qwen3.5/3.6, Qwen3-Next) select
**hybrid_stream**, and everything else (dense, except SWA which is excluded entirely)
selects **stream**. The decode side's derived-or-forced mode is what actually gets used
for a delegation - it is sent to the prefill server as `state_mode` in the
`POST /v1/prefill` request body,
and the prefill server honors it if its own model can serialize that way (whole mode
works for any non-SWA model; stream mode only for non-hybrid/non-recurrent models;
hybrid_stream only if the peer's own model also derives it - a plain non-SWA hybrid). A
request for a mode the peer cannot honor gets a protocol error and a clean local
fallback.

1. The decode server's slot computes its reusable local prefix (`n_past`) as usual (in
   whole and hybrid_stream mode this is still computed for bookkeeping, but does not
   affect eligibility or what gets transferred - see "When does a request get
   delegated?" above). If the
   request is eligible, the slot parks in a dedicated `SLOT_STATE_REMOTE_PREFILL` state -
   it joins no batches and is exempt from context shifting - while a worker thread
   performs the exchange. Other slots keep generating.
2. The prefill server turns the request into a normal completion-style task that stops
   after prompt processing.
   - **Stream mode**: after each decoded batch it serializes just that position range
     with `llama_state_seq_get_data_range()` and queues it to a writer thread; the
     inference loop never blocks on the network.
   - **Hybrid_stream mode**: identical to stream mode for the full-attention KV -
     `llama_state_seq_get_data_range()` per batch, streamed as `CHUNK` frames overlapped
     with compute. Then, once at prompt completion, it serializes only the recurrent
     state with `llama_state_seq_get_data_ext(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`
     and sends it as one `STATE` frame (`which = 2`, recurrent). The recurrent blob is
     O(1) in tokens (a few MB regardless of prompt length), so the whole attention KV is
     never materialized - peak host memory is one chunk plus that small blob. A `want_dft`
     draft blob is handled exactly as in whole mode.
   - **Whole mode**: each `update_slots` pass sends a `PROGRESS` message (just a token
     count, to keep the decode side's stream-timeout watchdog from firing during a long
     remote prefill) instead of chunks. At prompt completion it serializes the complete
     target-context state with `llama_state_seq_get_data_ext(..., flags=0)` and sends it
     as one `STATE` blob; if the decode side asked for the MTP draft context too
     (`want_dft`), the draft-context state is serialized and sent as a second `STATE`
     blob the same way. Both serializations run on the inference thread, same as
     stream-mode chunk serialization - it is a single long host-copy per blob (see
     Limitations).
3. The decode server applies the result on its inference thread.
   - **Stream mode**: each chunk is applied with
     `llama_state_seq_set_data_ext(..., LLAMA_STATE_SEQ_FLAGS_APPEND)` and the slot's
     token list is extended in lockstep, so the token list always mirrors KV contents.
   - **Hybrid_stream mode**: attention `CHUNK`s ride the exact same APPEND path as stream
     mode, building the attention KV and token list incrementally. At `DONE` the recurrent
     `STATE` blob is applied once with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (restoring only
     the recurrent part, leaving the chunk-built attention KV intact); the draft blob, if
     present, replaces the MTP sequence like whole mode. The slot stays in
     `SLOT_STATE_REMOTE_PREFILL` through every chunk and only flips to generating after the
     recurrent apply, so generation never starts on partial state (the resume barrier). Any
     failure drops the partial attention KV (all-or-nothing) and falls back to local prefill.
   - **Whole mode**: the target-context blob *replaces* the sequence
     (`llama_state_seq_set_data_ext(..., flags=0)`, not APPEND), and the draft-context
     blob (if present) replaces the MTP sequence the same way. This is all-or-nothing:
     if the draft apply fails after the target apply succeeded, both sequences are
     cleared and the request falls back to local prefill (no cooldown on this specific
     path - see Limitations). The slot's token list is rebuilt to mirror the restored
     state.
4. On completion the slot re-enters the normal prompt path. In whole and hybrid_stream
   mode the peer is sent all but the LAST prompt token, so the restored state ends one
   position short and the final token is decoded locally for logits with no rollback -
   rolling back a restored recurrent state is impossible and would otherwise force a
   full local reprocess, discarding the transfer. (In stream mode the last-token
   re-decode uses a one-cell KV rollback, which dense models support.) The local decode
   of the final token, in whole mode with MTP, exercises the restored draft context's
   speculation state. Generation proceeds; output is bit-identical to a local prefill.

Stale results from superseded delegation attempts are dropped via a per-attempt
generation id; peer-supplied ranges are bounds- and continuity-checked before use
(stream mode).

### Wire protocol (`POST /v1/prefill`)

**Request.** `POST /v1/prefill` with `Content-Type: application/json` and, if the decode
side has `--prefill-rpc-api-key`, `Authorization: Bearer KEY`. The body is flat JSON: the
compatibility descriptor plus the request fields.

```jsonc
{
  "proto":      3,               // protocol version, must match the peer
  "arch":       "qwen3",         // model arch + geometry (compat check)
  "n_layer":    64,
  "n_embd":     5120,
  "n_head_kv":  8,
  "n_ctx":      40960,
  "type_k":     1,               // ggml_type of the K / V cache
  "type_v":     1,
  "lora_hash":  "0",             // decimal string (uint64 is not JSON-number-safe)
  "has_dft":    false,           // decode side has an MTP draft context
  "state_mode": "stream",        // "stream", "whole", or "hybrid_stream" - mode picked for this request
  "want_dft":   false,           // optional; request the draft (MTP) state blob too
  "p0":         0,               // first position to stream back (stream mode)
  "tokens":     [1, 2, 3],       // the full prompt token ids
  "model":      "qwen3-30b"      // optional; ignored by the serving side
}
```

**Response (200).** `Content-Type: application/octet-stream`, a stream of length-prefixed
binary frames - `uint8 cmd | uint64 payload_size (LE) | payload` - with a 2 GiB per-frame
payload cap. Frame commands (`enum dp_cmd`):

```
# stream mode:
CHUNK    (2)  { u32 p0, u32 p1, u64 n_bytes, blob }   repeated, contiguous, ascending
DONE     (3)  { u32 n_tokens_total }                  success terminal
ERROR    (4)  { i32 code, u32 msg_len, msg }          failure terminal (msg not nul-terminated)

# whole mode:
PROGRESS (5)  { u32 n_processed }                     repeated keepalive, once per update_slots pass
STATE    (6)  { u8 which, u64 n_bytes, blob }         which: 0 = target, 1 = draft; target always
                                                      sent, draft iff want_dft
DONE     (3)  { u32 n_tokens_total }                  success terminal (or ERROR as above)

# hybrid_stream mode:
CHUNK    (2)  { u32 p0, u32 p1, u64 n_bytes, blob }   attention KV, repeated, contiguous, ascending
STATE    (6)  { u8 which, u64 n_bytes, blob }         which: 2 = recurrent (sent once at DONE),
                                                      1 = draft iff want_dft
DONE     (3)  { u32 n_tokens_total }                  success terminal (or ERROR as above)
```

The `which` byte on a `STATE` frame is `0` = whole target, `1` = draft (MTP), `2` =
recurrent tail (hybrid_stream). Values are wire-stable and append-only.

`DONE` is the only success signal; an `ERROR` frame, or the stream ending (EOF / read
timeout / connection drop) without a terminal frame, is a failure that triggers local
fallback. The decode side aborts the HTTP request (`httplib::Client::stop()`) to cancel.
The stream-mode (and hybrid_stream attention) chunk blob is exactly the output of
`llama_state_seq_get_data_range(ctx, ..., p0, p1, 0)`; the whole-mode state blob is
exactly the output of `llama_state_seq_get_data_ext(ctx, ..., seq_id, 0)` (unbounded, no
APPEND) for the target context, and the same call against the draft context for the
draft `STATE` frame. The hybrid_stream recurrent `STATE` blob (which=2) is the output of
`llama_state_seq_get_data_ext(ctx, ..., seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`.

**Pre-stream errors** are ordinary JSON HTTP error responses (`{"error": {...}}`), so the
status code is known before the body is interpreted as frames:

| Status | Cause |
|--------|-------|
| `400` | JSON parse failure, `proto`/compat mismatch, a mode the serving model cannot honor, `want_dft` with no serve-side draft, or a prompt longer than the serving context |
| `401` | api-key middleware rejected the request |
| `404` | `--prefill-serve` not set on the peer, so the route is never registered |
| `501` | `--prefill-serve` set but serving is inactive (e.g. an SWA model) |
| `503` | model still loading |

Failures that occur after the `200` header has been sent arrive in-band as an `ERROR`
frame instead.

### Core API additions (include/llama.h)

```c
size_t llama_state_seq_get_size_range(ctx, seq_id, p0, p1, flags);
size_t llama_state_seq_get_data_range(ctx, dst, size, seq_id, p0, p1, flags);
#define LLAMA_STATE_SEQ_FLAGS_APPEND 4  // restore without clearing the destination seq
```

Ranges are half-open `[p0, p1)`; `-1` means unbounded. APPEND restores must be applied
in ascending position order. Bounded ranges and APPEND are rejected (return 0) for SWA,
pure recurrent, and hybrid-iswa memory types and in combination with
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. A plain (non-SWA) hybrid forwards the range to its
full-attention KV cache and returns the attention-range size (hybrid_stream mode); its
recurrent part is not rangeable and is sent whole via the `PARTIAL_ONLY` flag. Whole mode
instead uses the existing unbounded, flags=0 `llama_state_seq_get/set_data_ext()` path -
the same one the server's prompt cache already uses for every architecture, including
hybrid/recurrent.

## MTP (speculative decoding with the same model)

A decode server with an MTP draft context (`--spec-type MTP`, i.e. a draft context
against the *same* model, as opposed to a separate `--model-draft`) can delegate in
whole or hybrid_stream mode (the target model determines which; a hybrid MTP model such
as Qwen3.6 uses hybrid_stream): the request carries `want_dft = 1`, and the prefill
server ships back both the target-context state and the draft-context state, applied
all-or-nothing. In both modes the draft context is a plain attention KV, so it is always
serialized and applied whole. This works because the serving slot's normal prompt
processing already populates the MTP draft context's KV via the existing MTP mirroring
hooks, so serializing `ctx_dft` at DONE_PROMPT is sufficient - nothing extra is computed
server-side just for delegation.

A separate `--model-draft` (classic speculative decoding with a different model) still
disables delegation entirely, in both modes: only same-model MTP contexts are paired,
and a mismatch (decode side requests `want_dft`, serve side has none) is rejected with a
pre-stream `400`. Stream mode still refuses to delegate at all when *any* draft context
(MTP or separate) is present on the decode side.

## Limitations

- SWA models: still excluded - SWA-masked cell serialization semantics are unresolved
  for both stream and whole mode; delegation disabled with a startup warning, local
  prefill used.
- No transfer/compute overlap in whole mode: the whole-sequence blob is only sent once
  prefill fully completes, unlike stream mode's per-batch chunks. Its serialization is a
  single long host-copy on the inference thread, which stalls other serving slots on that
  server for the duration - acceptable for a dedicated prefill box, but worth knowing
  about. Plain (non-SWA) hybrids avoid this: they now use hybrid_stream, so only the small
  recurrent tail is serialized whole; whole mode remains for pure recurrent models and
  forced (`--prefill-rpc-mode whole`) delegations. SWA / hybrid-iswa are excluded, not
  served via whole.
- hybrid_stream and stream overlap the attention KV with compute, but the last prompt
  token is still decoded locally (the restored recurrent/attention state ends one position
  short), so TTFT is remote-prefill + attention-transfer + one local decode.
- Both binaries must be the same protocol version (`proto` 3, this branch) - a mismatch
  is refused cleanly with a `400`, it does not silently negotiate down.
- A separate `--model-draft` (not MTP) still disables delegation entirely, in both
  modes.
- Whole-mode (and hybrid_stream recurrent-tail) apply failure has no cooldown (see the
  Failure handling table above) - a known gap, not expected to bite in practice.
- multimodal prompts, embedding/rerank tasks, alora requests: never delegated
- one prefill peer per decode server; no cross-peer load balancing
- IPv6 literals in `HOST:PORT` are not supported (hostnames and IPv4 work)
- Windows support is compiled in but untested

## Manual verification checklist (hybrid/MTP)

CI exercises stream and whole mode on the dense CI model (whole forced via
`--prefill-rpc-mode whole`) and hybrid_stream on a tiny hybrid (LFM2) fixture; there is no
tiny hybrid+MTP model available for automated testing, so the combination that matters
most for the target deployment - Qwen3.6-27B with MTP active - must be verified by hand
before relying on this in production:

1. Start both servers on separate boxes with the same Qwen3.6-27B-MTP build, decode side
   with `--spec-type MTP` active (and `--prefill-rpc gpubox:8080`), prefill side with
   `--host 0.0.0.0 --port 8080 --prefill-serve`.
2. Check the startup logs on **both** sides: each should log its derived mode as
   `hybrid_stream` (Qwen3.6 is a plain non-SWA hybrid, so both the decode-side
   "delegating prefill to ..." line and the serve-side "serving remote prefill over
   POST /v1/prefill (mode: ...)" line should reflect mode `hybrid_stream`, not `stream`
   or `whole`).
3. Send a long prompt (well above `--prefill-rpc-min-tokens`) to the decode server.
   Confirm the delegation log line shows `mode = hybrid_stream` and, since MTP is active,
   `(+dft)`; confirm the completion log shows the `A MiB attention + R MiB recurrent`
   split and a nonzero `dft:` byte count.
4. Compare output: send the identical prompt (same sampling params, greedy/temperature
   0) to a standalone (non-delegating) instance of the same model and confirm the
   completion text is byte-identical to the delegated run.
5. After the delegated response starts generating, verify MTP speculation is actually
   working post-resume: check the server's speculative-decoding acceptance-rate stats
   (or logs) are in the normal range for this model, not degraded - this is the
   least-proven part of a delegated MTP request (the prompt cache proves save/restore of
   draft state, but resuming *generation* with a restored draft state had not previously
   been exercised). If acceptance rates are noticeably worse than a standalone run, treat
   it as a regression before shipping.

## Testing

- `tests/test-state-seq-range.cpp` - C-level proof that chunked APPEND restore is
  bitwise-identical to whole-blob restore (`ctest -R test-state-seq-range`, needs
  `LLAMACPP_TEST_MODELFILE` or a model path argument)
- `tools/server/tests/unit/test_prefill_rpc.py` - integration tests: protocol
  conformance (`POST /v1/prefill` request/descriptor validation, chunk
  contiguity/coverage, whole-mode PROGRESS/STATE/DONE framing, hybrid_stream
  CHUNK...STATE(which=2)/DONE framing on a tiny LFM2 hybrid, pre-stream HTTP error
  codes), byte-identical delegated vs local output in stream, whole, and hybrid_stream
  mode (including hybrid_stream == whole for the same hybrid model), metrics, dead-peer
  fallback, peer death between requests, stalling-peer timeout, bad/unhonorable mode
  negotiation, `want_dft` without a draft context, and the cache_prompt-off regression
