import os
import re
import socket
import struct
import tempfile
import threading
import time
import pytest
import requests
from utils import *

server: ServerProcess

# streamed-body frame commands (server-prefill.h dp_cmd); wire-stable
DP_CMD_CHUNK, DP_CMD_DONE, DP_CMD_ERROR = 2, 3, 4
DP_CMD_PROGRESS, DP_CMD_STATE = 5, 6
DP_PROTO_VERSION = 3  # server-prefill.h DP_PROTO_VERSION

# port used only by the stall test's raw accept-and-hang socket (the serving
# side now lives on the main HTTP port, so there is no separate dp listener)
STALL_PORT = 8199


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # the tinyllama2 tokenizer produces ~35 tokens per repeat of the test
    # sentence (281 tokens for the 8x prompt below); bump the context size
    # so the whole prompt fits in a single serving slot's KV cache
    # (n_ctx_slot = n_ctx / n_parallel with the default n_slots = 2).
    server.n_ctx = 2048
    server.prefill_serve = True


def metric_value(metrics_text, name):
    m = re.search(re.escape(name) + r" ([0-9.eE+-]+)", metrics_text)
    assert m, f"metric {name} not found"
    return float(m.group(1))


class LogReader:
    # incremental reader over a ServerProcess log_path file (same pattern as
    # test_kv_keep_only_active.py)
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


# --- HTTP data-plane helpers: build POST /v1/prefill requests, parse the framed body ---

def props_prefill(srv):
    """The serving descriptor. It lives in the /v1/models item's "prefill"
    object (a public, no-auth endpoint), exposing exactly the fields a
    compatible request body must echo back."""
    models = srv.make_request("GET", "/v1/models").body
    desc = models["data"][0]["prefill"]
    assert desc["enabled"] is True
    return desc


def prefill_body(srv, tokens, p0=0, mode="stream", want_dft=False, **overrides):
    """Build a flat POST /v1/prefill body from the serving descriptor so the
    compatibility check passes. Pass overrides to force negative-test values."""
    d = props_prefill(srv)
    body = {
        "proto":      DP_PROTO_VERSION,
        "arch":       d["arch"],
        "n_layer":    d["n_layer"],
        "n_embd":     d["n_embd"],
        "n_head_kv":  d["n_head_kv"],
        "n_ctx":      d["n_ctx"],
        "type_k":     d["type_k"],
        "type_v":     d["type_v"],
        "lora_hash":  "0",  # decimal string; "0" matches the serving side (warning-only anyway)
        "has_dft":    d["has_dft"],
        "state_mode": mode,
        "want_dft":   want_dft,
        "p0":         p0,
        "tokens":     tokens,
    }
    body.update(overrides)
    return body


def prefill_post(srv, body, api_key=None, stream=True, timeout=60):
    url = f"http://{srv.server_host}:{srv.server_port}/v1/prefill"
    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, json=body, headers=headers or None, stream=stream, timeout=timeout)


def iter_frames(resp):
    """Drain a streamed application/octet-stream response, returning a list of
    (cmd, payload) frames. Framing: 1-byte cmd + 8-byte LE uint64 size + payload."""
    frames = []
    buf = b""
    have_header = False
    cmd = 0
    size = 0
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        buf += chunk
        while True:
            if not have_header:
                if len(buf) < 9:
                    break
                cmd, size = struct.unpack("<BQ", buf[:9])
                buf = buf[9:]
                have_header = True
            if len(buf) < size:
                break
            frames.append((cmd, buf[:size]))
            buf = buf[size:]
            have_header = False
    return frames


def tokenize_long_text(srv, api_key=None):
    text = " ".join(["hello world, the quick brown fox jumps over the lazy dog"] * 8)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    res = srv.make_request("POST", "/tokenize", data={"content": text, "add_special": True}, headers=headers)
    tokens = res.body["tokens"]
    assert len(tokens) > 64
    return tokens


# --- serving-side direct-POST tests -----------------------------------------

def test_prefill_serve_streams_chunks():
    global server
    server.start()

    tokens = tokenize_long_text(server)
    resp = prefill_post(server, prefill_body(server, tokens))
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("application/octet-stream")

    frames = iter_frames(resp)

    got_done = False
    next_p0 = 0
    total_bytes = 0
    for cmd, payload in frames:
        if cmd == DP_CMD_CHUNK:
            c_p0, c_p1, n_bytes = struct.unpack("<IIQ", payload[:16])
            assert c_p0 == next_p0, "chunks must be contiguous and ordered"
            assert c_p1 > c_p0
            assert len(payload) == 16 + n_bytes
            next_p0 = c_p1
            total_bytes += n_bytes
        elif cmd == DP_CMD_DONE:
            (n_total,) = struct.unpack("<I", payload)
            assert n_total == len(tokens)
            got_done = True
        else:
            pytest.fail(f"unexpected cmd {cmd}")

    assert got_done, "stream must end with a DONE frame"
    assert next_p0 == len(tokens), "chunks must cover the requested range"
    assert total_bytes > 0

    # the serving slot must be released and usable afterwards
    res = server.make_request("POST", "/completion", data={"prompt": "hello", "n_predict": 4})
    assert res.status_code == 200


def test_prefill_serve_whole_mode():
    global server
    server.start()
    tokens = tokenize_long_text(server)
    resp = prefill_post(server, prefill_body(server, tokens, mode="whole"))
    assert resp.status_code == 200

    got_state = got_done = False
    saw_progress = False
    for cmd, payload in iter_frames(resp):
        if cmd == DP_CMD_PROGRESS:
            (n,) = struct.unpack("<I", payload)
            assert 0 < n <= len(tokens)
            saw_progress = True
        elif cmd == DP_CMD_STATE:
            which, n_bytes = struct.unpack("<BQ", payload[:9])
            assert which == 0 and not got_state  # main blob exactly once, no dft requested
            assert len(payload) == 9 + n_bytes and n_bytes > 0
            got_state = True
        elif cmd == DP_CMD_DONE:
            assert got_state
            got_done = True
        else:
            pytest.fail(f"unexpected cmd {cmd}")
    assert got_done and got_state
    # PROGRESS is emitted to keep the read timeout fed; at least one is expected
    assert saw_progress

    res = server.make_request("POST", "/completion", data={"prompt": "hello", "n_predict": 4})
    assert res.status_code == 200


def test_prefill_serve_rejects_bad_version():
    global server
    server.start()
    resp = prefill_post(server, prefill_body(server, [1, 2, 3, 4], proto=99))
    assert resp.status_code == 400
    # dp_prefill_request_parse's exact reason string for a proto mismatch
    assert "protocol version mismatch" in resp.json()["error"]["message"]


def test_prefill_serve_rejects_compat_mismatch():
    # NEW: a request whose descriptor disagrees with the serving model must be
    # rejected with a message that names the mismatched field
    global server
    server.start()
    d = props_prefill(server)
    resp = prefill_post(server, prefill_body(server, [1, 2, 3, 4], n_layer=d["n_layer"] + 1))
    assert resp.status_code == 400
    assert "n_layer" in resp.json()["error"]["message"]


def test_prefill_serve_rejects_too_long():
    global server
    server.start()
    n = 100000  # far beyond the serving context size
    resp = prefill_post(server, prefill_body(server, [1] * n))
    assert resp.status_code == 400
    assert "context" in resp.json()["error"]["message"].lower()


def test_prefill_serve_rejects_bad_mode():
    global server
    server.start()
    resp = prefill_post(server, prefill_body(server, [1, 2, 3, 4], state_mode="bogus"))
    assert resp.status_code == 400
    assert "state_mode" in resp.json()["error"]["message"]


def test_prefill_serve_rejects_want_dft():
    # the CI model has no draft/MTP context, so want_dft must be rejected
    global server
    server.start()
    resp = prefill_post(server, prefill_body(server, [1, 2, 3, 4], want_dft=True))
    assert resp.status_code == 400
    # the post_prefill handler's exact rejection for want_dft without a draft
    assert "draft state not available" in resp.json()["error"]["message"]


def test_prefill_serve_404_when_disabled():
    # NEW: without --prefill-serve the route is not registered at all, so the
    # body is irrelevant - the router-less server simply has no such endpoint
    global server
    server.prefill_serve = None
    server.start()
    # sanity: the descriptor advertises prefill as disabled
    desc = server.make_request("GET", "/v1/models").body["data"][0]["prefill"]
    assert desc["enabled"] is False

    resp = prefill_post(server, {"proto": DP_PROTO_VERSION, "tokens": [1, 2, 3, 4]})
    assert resp.status_code == 404


def test_prefill_serve_api_key():
    # NEW: POST /v1/prefill is behind the API key; bare -> 401, Bearer -> 200
    global server
    server.api_key = "sk-prefill-serve-secret"
    server.start()

    tokens = tokenize_long_text(server, api_key=server.api_key)
    body = prefill_body(server, tokens)  # /v1/models is public, so this needs no auth

    unauth = prefill_post(server, body, api_key=None)
    assert unauth.status_code == 401

    ok = prefill_post(server, body, api_key=server.api_key)
    assert ok.status_code == 200
    cmds = [cmd for cmd, _ in iter_frames(ok)]
    assert DP_CMD_DONE in cmds


def read_one_frame(resp):
    """Incrementally read a streamed response until exactly one complete frame
    (cmd, payload) has arrived, leaving the rest of the stream unread."""
    buf = b""
    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf += chunk
        if len(buf) < 9:
            continue
        cmd, size = struct.unpack("<BQ", buf[:9])
        if len(buf) >= 9 + size:
            return cmd, buf[9:9 + size]
    raise AssertionError("stream ended before a complete frame arrived")


def test_prefill_client_disconnect_cancels():
    # NEW: a client that starts a prefill and disconnects mid-stream must not
    # wedge the serving slot - the response destructor's close_read() makes the
    # queue thread's emit_prefill_chunks() see a dead pipe and release the slot.
    # to make the disconnect genuinely in-flight (and not a no-op after the
    # prefill already finished), the serving batch is tiny and the prompt long,
    # so the stream spans hundreds of CHUNKs; we read exactly ONE and hang up.
    global server
    server.n_slots = 1   # a single slot makes the follow-up completion depend on the release
    server.n_batch = 4
    server.n_ubatch = 4
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()
    log = LogReader(server.log_path)

    text = " ".join(["hello world, the quick brown fox jumps over the lazy dog"] * 30)
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    tokens = res.body["tokens"]
    assert len(tokens) > 800  # / n_batch 4 -> hundreds of CHUNK frames

    resp = prefill_post(server, prefill_body(server, tokens))
    assert resp.status_code == 200
    cmd, _ = read_one_frame(resp)
    assert cmd == DP_CMD_CHUNK
    # abruptly drop the connection with the vast majority of the stream unread
    resp.close()

    # serve-side evidence the abort path actually ran: emit_prefill_chunks()
    # logs this exact line (server-context.cpp) when it finds the pipe dead
    # mid-prefill and releases the slot. if the prefill had already finished,
    # the slot would be released via the normal DONE path and never log this.
    deadline = time.time() + 20
    log_text = ""
    while time.time() < deadline:
        log_text += log.drain()
        if "prefill client disconnected, releasing slot" in log_text:
            break
        time.sleep(0.25)
    assert "prefill client disconnected, releasing slot" in log_text, \
        "serve side never took the dead-pipe release path (disconnect was not in-flight?)"

    # and the (only) serving slot must free promptly: a follow-up completion succeeds
    deadline = time.time() + 30
    ok = False
    while time.time() < deadline:
        res = server.make_request("POST", "/completion", data={"prompt": "hello", "n_predict": 4})
        if res.status_code == 200:
            ok = True
            break
        time.sleep(0.5)
    assert ok, "serving slot never freed after client disconnect"


# --- delegated (decode-side) tests ------------------------------------------

LONG_TEXT = " ".join(["the quick brown fox jumps over the lazy dog while the cat watches"] * 12)
# a second, unrelated long prompt: used where a test needs a prompt that does
# NOT share a cached prefix with LONG_TEXT (see test_fallback_when_peer_dies_mid_service)
LONG_TEXT_2 = " ".join(["a distant lighthouse guides the sailors home through heavy fog"] * 12)
GREEDY = {"n_predict": 24, "temperature": 0.0, "seed": 42, "cache_prompt": True}


def test_delegated_prefill_matches_local():
    global server
    # baseline: the prefill server itself answers the request (same model/config)
    server.start()
    baseline = server.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
    assert baseline.status_code == 200

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8091
    decode.n_ctx = 2048
    decode.server_metrics = True  # exposes /metrics
    decode.prefill_rpc = f"127.0.0.1:{server.server_port}"
    decode.prefill_rpc_min_tokens = 8
    decode.start()
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 1" in metrics
        # prove the remote chunks were actually applied, not that every chunk
        # failed and the request quietly completed via local fallback
        assert "llamacpp:prefill_fallback_total 0" in metrics

        # bandwidth/throughput counters: values vary with tokenizer output and
        # blob encoding, so parse and check magnitude rather than matching a
        # hardcoded number
        assert metric_value(metrics, "llamacpp:prefill_delegated_tokens_total") > 0
        assert metric_value(metrics, "llamacpp:prefill_rpc_bytes_total") > 0
    finally:
        decode.stop()


def test_delegated_prefill_without_cache_prompt():
    # regression: cache_prompt=false must not discard the applied remote KV and
    # re-delegate the same task forever (delegate -> discard -> delegate loop)
    global server
    server.start()
    baseline = server.make_request("POST", "/completion",
                                   data={"prompt": LONG_TEXT, **GREEDY, "cache_prompt": False})
    assert baseline.status_code == 200

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8092
    decode.n_ctx = 2048
    decode.server_metrics = True
    decode.prefill_rpc = f"127.0.0.1:{server.server_port}"
    decode.prefill_rpc_min_tokens = 8
    decode.start()
    try:
        res = decode.make_request("POST", "/completion",
                                  data={"prompt": LONG_TEXT, **GREEDY, "cache_prompt": False})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1  # exactly once, no loop
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
    finally:
        decode.stop()


def test_delegated_prefill_whole_mode_matches_local():
    # forces whole mode on the dense CI model; serve side honors it (capability-based)
    global server
    server.start()
    baseline = server.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
    assert baseline.status_code == 200

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8093
    decode.n_ctx = 2048
    decode.server_metrics = True
    decode.prefill_rpc = f"127.0.0.1:{server.server_port}"
    decode.prefill_rpc_min_tokens = 8
    decode.prefill_rpc_mode = "whole"
    decode.start()
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
        assert metric_value(metrics, "llamacpp:prefill_rpc_bytes_total") > 0
    finally:
        decode.stop()


def test_delegated_prefill_whole_mode_without_cache_prompt():
    # the loop-guard must hold in whole mode too: cache_prompt=false must not
    # discard the applied remote whole-blob state and re-delegate the same task
    # forever (delegate -> discard -> delegate loop)
    global server
    server.start()
    baseline = server.make_request("POST", "/completion",
                                   data={"prompt": LONG_TEXT, **GREEDY, "cache_prompt": False})
    assert baseline.status_code == 200

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8094
    decode.n_ctx = 2048
    decode.server_metrics = True
    decode.prefill_rpc = f"127.0.0.1:{server.server_port}"
    decode.prefill_rpc_min_tokens = 8
    decode.prefill_rpc_mode = "whole"
    decode.start()
    try:
        res = decode.make_request("POST", "/completion",
                                  data={"prompt": LONG_TEXT, **GREEDY, "cache_prompt": False})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1  # exactly once, no loop
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
    finally:
        decode.stop()


# --- fallback tests ---------------------------------------------------------

def test_fallback_when_peer_is_down():
    # decode server pointed at a dead port: first long request must still succeed
    decode = ServerPreset.tinyllama2()
    decode.server_port = 8095
    decode.n_ctx = 2048  # the whole LONG_TEXT prompt must fit in a single slot's KV cache
    decode.server_metrics = True
    decode.prefill_rpc = "127.0.0.1:9"  # discard port, nothing listens
    decode.prefill_rpc_min_tokens = 8
    decode.start()
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
        assert res.status_code == 200
        assert len(res.body["content"]) > 0

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 1" in metrics
        assert "llamacpp:prefill_fallback_total 1" in metrics

        # immediate second request, with a large *uncached* suffix (well past
        # prefill_rpc_min_tokens) appended after the already-cached LONG_TEXT
        # prefix: if the 60s cooldown were not honored, this would attempt a
        # second delegation to the dead peer. it must not - the request still
        # succeeds (fallback is invisible), delegated_total must NOT increase
        # (cooldown skipped the attempt entirely), and fallback_total must NOT
        # increase either: a request that was never delegated is not a "failed
        # delegation" and must not be double-counted as a fallback.
        res = decode.make_request(
            "POST", "/completion", data={"prompt": LONG_TEXT + " " + LONG_TEXT, **GREEDY}
        )
        assert res.status_code == 200
        assert len(res.body["content"]) > 0

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 1" in metrics
        assert "llamacpp:prefill_fallback_total 1" in metrics
    finally:
        decode.stop()


def test_fallback_when_peer_dies_mid_service():
    global server
    server.start()  # prefill server

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8096
    decode.n_ctx = 2048
    decode.server_metrics = True
    decode.prefill_rpc = f"127.0.0.1:{server.server_port}"
    decode.prefill_rpc_min_tokens = 8
    decode.start()
    try:
        # first delegated request works
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY})
        assert res.status_code == 200
        assert len(res.body["content"]) > 0

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 1" in metrics
        assert "llamacpp:prefill_fallback_total 0" in metrics

        # kill the prefill server, then send a *different* long prompt. reusing
        # LONG_TEXT here would be vacuous: the decode server's own prompt cache
        # already covers the whole prompt after request 1, so n_past would
        # cover the entire prompt and delegation would never even be attempted
        # - the request would trivially succeed without exercising the fallback
        # path at all. LONG_TEXT_2 shares no cached prefix, so it forces a
        # fresh delegation attempt against the now-dead peer.
        server.stop()
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT_2, **GREEDY})
        assert res.status_code == 200
        assert len(res.body["content"]) > 0

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 2" in metrics
        assert "llamacpp:prefill_fallback_total 1" in metrics
    finally:
        decode.stop()


def test_fallback_when_peer_stalls():
    # a peer that accepts the TCP connection but never sends any HTTP response
    # must not hang the delegating request forever: the stream-phase per-recv
    # read timeout (LLAMA_PREFILL_READ_TIMEOUT_MS) must fire, failing the
    # delegation and falling back to local prefill. This is a plain socket
    # listener, not a real prefill peer, so it never speaks HTTP at all - it
    # just accepts, reads and discards whatever arrives, and stays silent.
    stall_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stall_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stall_sock.bind(("127.0.0.1", STALL_PORT))
    stall_sock.listen(1)
    stop_stalling = threading.Event()

    def stall_listener():
        try:
            stall_sock.settimeout(35)
            conn, _ = stall_sock.accept()
        except OSError:
            return
        try:
            conn.settimeout(1)
            deadline = time.time() + 30
            while time.time() < deadline and not stop_stalling.is_set():
                try:
                    if not conn.recv(4096):
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            conn.close()

    listener_thread = threading.Thread(target=stall_listener, daemon=True)
    listener_thread.start()

    decode = ServerPreset.tinyllama2()
    decode.server_port = 8097
    decode.n_ctx = 2048
    decode.server_metrics = True
    decode.prefill_rpc = f"127.0.0.1:{STALL_PORT}"
    decode.prefill_rpc_min_tokens = 8
    # small per-recv read timeout so a silent peer fails fast instead of the
    # 120s default; the elapsed-time assertion below is what proves it fired
    decode.extra_env = {"LLAMA_PREFILL_READ_TIMEOUT_MS": 8000}
    decode.start()
    try:
        start = time.time()
        # generous request timeout so the test itself never times out the HTTP
        # client first; the elapsed assertion is what proves the read timeout
        # applied rather than some much longer bound.
        res = decode.make_request("POST", "/completion", data={"prompt": LONG_TEXT, **GREEDY}, timeout=60)
        elapsed = time.time() - start
        assert res.status_code == 200
        assert len(res.body["content"]) > 0
        # well under the 120s default read timeout (and the request's own 60s
        # client-side timeout): proves the 8s read timeout is what fired.
        assert elapsed < 30, "request took long enough to suggest the default read timeout (or longer) applied"

        metrics = decode.make_request("GET", "/metrics").body
        assert "llamacpp:prefill_delegated_total 1" in metrics
        assert "llamacpp:prefill_fallback_total 1" in metrics
    finally:
        decode.stop()
        stop_stalling.set()
        stall_sock.close()
        listener_thread.join(timeout=5)


# --- hybrid-stream (plain-hybrid model) tests -------------------------------
# LFM2-test-ci-80M is llama_model_is_hybrid and non-SWA, so both the serving and
# decode servers derive state_mode == hybrid_stream. Attention KV streams as
# position-range CHUNK frames; the bounded recurrent state is sent once at the
# end as STATE(which=2) (server-prefill.h DP_STATE_TARGET_RECURRENT).

DP_STATE_TARGET_RECURRENT = 2  # server-prefill.h dp_state_which; recurrent tail id

# ~240 LFM2 tokens (> n_batch 64), so the attention KV spans multiple CHUNKs
HYBRID_TEXT = " ".join(["the quick brown fox jumps over the lazy dog while the cat watches"] * 12)
# a longer prompt (~960 tokens) whose whole-mode blob would be large
HYBRID_TEXT_LONG = " ".join(["the quick brown fox jumps over the lazy dog while the cat watches"] * 48)
# an unrelated prompt sharing no cached prefix with HYBRID_TEXT (fallback test)
HYBRID_TEXT_2 = " ".join(["a distant lighthouse guides the sailors home through heavy fog"] * 12)


def hybrid_peer():
    peer = ServerPreset.hybrid_lfm2()
    peer.prefill_serve = True
    return peer


def hybrid_decode(port, **overrides):
    decode = ServerPreset.hybrid_lfm2()
    decode.server_port = port
    decode.server_metrics = True
    decode.prefill_rpc_min_tokens = 8
    for k, v in overrides.items():
        setattr(decode, k, v)
    return decode


def hybrid_tokens(srv, text=HYBRID_TEXT):
    res = srv.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    tokens = res.body["tokens"]
    assert len(tokens) > 64  # > n_batch, so the attention KV streams as multiple CHUNKs
    return tokens


def test_hybrid_stream_matches_local():
    peer = hybrid_peer()
    peer.start()
    # the served model must derive the streaming hybrid mode, not whole
    assert props_prefill(peer)["mode"] == "hybrid_stream"

    baseline = peer.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT, **GREEDY})
    assert baseline.status_code == 200

    decode = hybrid_decode(8101, prefill_rpc=f"127.0.0.1:{peer.server_port}")
    decode.start()
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
        assert metric_value(metrics, "llamacpp:prefill_rpc_bytes_total") > 0
    finally:
        decode.stop()

    # the on-wire split: several attention CHUNKs then exactly one recurrent tail.
    # use a prompt the peer has NOT already served: a fully-cached prompt streams
    # as a single [0,n) range, collapsing the multi-chunk assertion below.
    tokens = hybrid_tokens(peer, HYBRID_TEXT_2)
    resp = prefill_post(peer, prefill_body(peer, tokens, mode="hybrid_stream"))
    assert resp.status_code == 200
    n_chunks = n_states = 0
    state_which = -1
    got_done = False
    next_p0 = 0
    for cmd, payload in iter_frames(resp):
        if cmd == DP_CMD_CHUNK:
            c_p0, c_p1, n_bytes = struct.unpack("<IIQ", payload[:16])
            assert c_p0 == next_p0 and c_p1 > c_p0
            next_p0 = c_p1
            n_chunks += 1
        elif cmd == DP_CMD_STATE:
            state_which, n_bytes = struct.unpack("<BQ", payload[:9])
            assert n_bytes > 0 and not got_done  # recurrent tail comes before DONE
            n_states += 1
        elif cmd == DP_CMD_DONE:
            got_done = True
    assert n_chunks > 1, "prompt must span multiple attention chunks"
    assert n_states == 1 and state_which == DP_STATE_TARGET_RECURRENT
    assert got_done


def test_hybrid_stream_fallback():
    peer = hybrid_peer()
    peer.start()

    decode = hybrid_decode(8102, prefill_rpc=f"127.0.0.1:{peer.server_port}")
    decode.start()
    try:
        # first delegated request works
        res = decode.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT, **GREEDY})
        assert res.status_code == 200
        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0

        # capture the correct answer for the second prompt while the peer is alive
        baseline2 = peer.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT_2, **GREEDY})
        assert baseline2.status_code == 200

        # kill the serving peer, then send a *different* long prompt: HYBRID_TEXT
        # is already cached on decode, so reusing it would never re-delegate.
        peer.stop()
        res = decode.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT_2, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline2.body["content"]  # correct via local fallback

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 2
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 1
    finally:
        decode.stop()


def test_hybrid_stream_completes_long_prompt():
    # bounded-memory stand-in: a prompt whose whole-mode blob would be large
    # streams as bounded attention CHUNKs + one O(1)-in-tokens recurrent tail and
    # completes without tripping a "state too large" guard. LFM2-80M is tiny, so
    # this only asserts a correct completion in hybrid_stream mode over a long
    # prompt; the real bounded-peak-memory proof is the two-box 35B gate
    # (H-plan verification step 3), not an RSS sample here.
    peer = hybrid_peer()
    peer.start()
    baseline = peer.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT_LONG, **GREEDY})
    assert baseline.status_code == 200

    decode = hybrid_decode(8103, prefill_rpc=f"127.0.0.1:{peer.server_port}")
    decode.start()
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT_LONG, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
    finally:
        decode.stop()


def test_hybrid_stream_matches_whole():
    # forcing --prefill-rpc-mode whole on the decode side makes the same hybrid
    # model serve its state as a single whole blob; greedy output must be
    # identical to both the hybrid_stream delegation and the standalone answer,
    # proving the two transfer modes are equivalent for a hybrid model.
    peer = hybrid_peer()
    peer.start()
    baseline = peer.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT, **GREEDY})
    assert baseline.status_code == 200

    decode = hybrid_decode(8104, prefill_rpc=f"127.0.0.1:{peer.server_port}", prefill_rpc_mode="whole")
    fd, decode.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    decode.start()
    log = LogReader(decode.log_path)
    try:
        res = decode.make_request("POST", "/completion", data={"prompt": HYBRID_TEXT, **GREEDY})
        assert res.status_code == 200
        assert res.body["content"] == baseline.body["content"]

        metrics = decode.make_request("GET", "/metrics").body
        assert metric_value(metrics, "llamacpp:prefill_delegated_total") == 1
        assert metric_value(metrics, "llamacpp:prefill_fallback_total") == 0
    finally:
        decode.stop()

    # prove the override actually took effect: the decode side logs the exact
    # transfer mode it delegated in (server-context.cpp "delegating prefill,
    # ... mode = %s"). without this the equality assertion above would still
    # pass if the override were silently ignored and hybrid_stream ran instead.
    log_text = log.drain()
    assert "mode = whole" in log_text, "decode did not delegate in whole mode"
    assert "mode = hybrid_stream" not in log_text
