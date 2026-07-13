#include "server-prefill.h"

#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>

// json (nlohmann::ordered_json) comes from server-prefill.h; used by the HTTP
// prefill protocol helpers.

// per-recv read timeout on the streaming POST /v1/prefill (in seconds/useconds
// for httplib). frames and PROGRESS records keep it fed, so it only fires on
// genuine mid-stream silence - same semantics as the old SO_RCVTIMEO. bump
// LLAMA_PREFILL_READ_TIMEOUT_MS for very large/slow remote prefills.
static void dp_read_timeout(int & secs, int & usecs) {
    int ms = 120000;
    if (const char * s = std::getenv("LLAMA_PREFILL_READ_TIMEOUT_MS")) {
        char *     end = nullptr;
        const long v   = strtol(s, &end, 10);
        if (end != s && v > 0) {
            ms = (int) v;
        }
    }
    secs  = ms / 1000;
    usecs = (ms % 1000) * 1000;
}

// --- server_prefill_client ---

bool server_prefill_client::init(const std::string & host_, int port_, const std::string & api_key_,
                                  const dp_hello & local_hello, int max_inflight_,
                                  chunk_cb on_chunk_, done_cb on_done_, whole_cb on_whole_,
                                  recurrent_cb on_recurrent_) {
    host         = host_;
    port         = port_;
    api_key      = api_key_;
    hello_local  = local_hello;
    max_inflight = max_inflight_ > 0 ? max_inflight_ : 1;
    on_chunk     = std::move(on_chunk_);
    on_done      = std::move(on_done_);
    on_whole     = std::move(on_whole_);
    on_recurrent = std::move(on_recurrent_);
    return true;
}

// must be called with mtx held; drops workers whose thread body has already
// finished (never joins a thread that is still running)
void server_prefill_client::reap_finished() {
    for (auto it = workers.begin(); it != workers.end();) {
        if (it->done->load()) {
            it->thread.join();
            it = workers.erase(it);
        } else {
            ++it;
        }
    }
}

bool server_prefill_client::try_delegate(int id_slot, int id_task, std::vector<llama_token> && tokens, uint32_t p0,
                                          dp_state_mode mode, bool want_dft, uint64_t & out_gen) {
    std::lock_guard<std::mutex> lk(mtx);

    reap_finished();

    if (clients_by_slot.count(id_slot)) {
        return false; // a slot with a live worker cannot be re-delegated
    }

    if (ggml_time_us() < t_cooldown_until_us) {
        return false;
    }

    if (n_inflight >= max_inflight) {
        return false;
    }
    ++n_inflight;

    const uint64_t gen = generation_next++;

    auto done_flag = std::make_shared<std::atomic<bool>>(false);

    const std::string  host_local = host;
    const int          port_local = port;
    const std::string  api_key_local = api_key;
    const dp_hello     hello      = hello_local;
    chunk_cb           cb_chunk   = on_chunk;
    done_cb            cb_done    = on_done;
    whole_cb           cb_whole   = on_whole;
    recurrent_cb       cb_recur   = on_recurrent;

    std::thread th([this, id_slot, id_task, gen, tokens = std::move(tokens), p0, mode, want_dft,
                     host_local, port_local, api_key_local, hello, cb_chunk, cb_done, cb_whole, cb_recur, done_flag]() {
        // one client for this worker's whole lifetime. registered in
        // clients_by_slot so cancel()/stop() can call stop() on it from another
        // thread while send() blocks - httplib's documented thread-safe
        // abort. entries are erased only if still pointing at &cli, so a stale
        // worker never drops a newer worker's entry for the same slot.
        httplib::Client cli(host_local, port_local);
        {
            std::lock_guard<std::mutex> lk2(mtx);
            clients_by_slot[id_slot] = &cli;
        }

        // this worker reserved its inflight slot synchronously in try_delegate()
        // and always owns it.
        bool owns_inflight = true;

        // erases clients_by_slot and sets *done_flag last, taking no lock after
        // it - reap_finished() joins done workers under mtx. cancelled_slots is
        // about suppressing the cooldown on user cancel.
        auto fail = [&](const std::string & msg, bool no_cooldown = false) {
            {
                std::lock_guard<std::mutex> lk2(mtx);
                const bool was_cancelled = cancelled_slots.erase(id_slot) > 0;
                if (!was_cancelled && !no_cooldown) {
                    t_cooldown_until_us = ggml_time_us() + 60ll * 1000 * 1000; // 60 s
                }
                auto it = clients_by_slot.find(id_slot);
                if (it != clients_by_slot.end() && it->second == &cli) {
                    clients_by_slot.erase(it);
                }
                if (owns_inflight) {
                    --n_inflight;
                }
            }
            cb_done(id_slot, id_task, gen, false, msg);
            *done_flag = true;
        };

        // clears cooldown and drops this worker's inflight/client bookkeeping on
        // a successful terminal (DONE); like fail(), takes no lock afterwards.
        auto finish_ok = [&]() {
            std::lock_guard<std::mutex> lk2(mtx);
            cancelled_slots.erase(id_slot);
            auto it = clients_by_slot.find(id_slot);
            if (it != clients_by_slot.end() && it->second == &cli) {
                clients_by_slot.erase(it);
            }
            if (owns_inflight) {
                --n_inflight;
            }
            t_cooldown_until_us = 0;
        };

        // guard against an unhandled exception (e.g. bad_alloc from an
        // oversized payload) reaching std::thread's entry point and calling
        // std::terminate on the whole process. *done_flag is only ever set true
        // right before a fail()/success return below, so checking it here tells
        // the catch whether the terminal bookkeeping has already run -
        // guarantees it happens exactly once.
        try {
            // stream phase: POST the request and feed the framed response body
            // to the parser. the read timeout is per-recv (frames/PROGRESS keep
            // it fed), so it only fires on genuine mid-stream silence.
            cli.set_connection_timeout(5, 0);
            int rt_s = 0, rt_us = 0;
            dp_read_timeout(rt_s, rt_us);
            cli.set_read_timeout(rt_s, rt_us);
            cli.set_write_timeout(30, 0);

            // flat request body: the local hello at top level, plus the request
            // fields; state_mode is overwritten with the caller's resolved mode.
            json body = dp_hello_to_json(hello);
            body["state_mode"] = dp_state_mode_str(mode);
            body["want_dft"]   = want_dft;
            body["p0"]         = p0;
            body["tokens"]     = tokens;

            // whole and hybrid-stream: STATE blobs collected here, applied
            // atomically once DONE confirms both arrived. whole mode's "main" is
            // the whole target blob (which=0); hybrid-stream's is the recurrent
            // tail (which=2), applied after the attention CHUNKs. the draft
            // (which=1) is optional in both.
            bool                 have_state_main = false;
            bool                 have_state_dft  = false;
            std::vector<uint8_t> state_main;
            std::vector<uint8_t> state_dft;

            dp_frame_parser parser;
            parser.on_record = [&](dp_cmd cmd, std::vector<uint8_t> && payload) -> bool {
                if (cmd == DP_CMD_CHUNK) {
                    if (payload.size() < sizeof(dp_msg_chunk_hdr)) { fail("malformed chunk"); return false; }
                    dp_msg_chunk_hdr ch; memcpy(&ch, payload.data(), sizeof(ch));
                    if (payload.size() != sizeof(ch) + ch.n_bytes) { fail("malformed chunk"); return false; }
                    std::vector<uint8_t> blob(payload.begin() + sizeof(ch), payload.end());
                    cb_chunk(id_slot, id_task, gen, ch.p0, ch.p1, std::move(blob));
                    return true;
                } else if (cmd == DP_CMD_PROGRESS) {
                    // whole mode only: keeps the read timeout from firing during
                    // a long remote prefill; nothing to apply yet
                    dp_msg_progress pr = {};
                    if (payload.size() >= sizeof(pr)) {
                        memcpy(&pr, payload.data(), sizeof(pr));
                    }
                    LOG_DBG("prefill progress: %u tokens processed\n", pr.n_processed);
                    return true;
                } else if (cmd == DP_CMD_STATE) {
                    if (payload.size() < sizeof(dp_msg_state_hdr)) { fail("malformed state frame"); return false; }
                    dp_msg_state_hdr sh; memcpy(&sh, payload.data(), sizeof(sh));
                    if (payload.size() != sizeof(sh) + sh.n_bytes) { fail("malformed state frame"); return false; }
                    // route by mode: whole => which 0 is the target blob; hybrid-stream
                    // => which 2 is the recurrent tail. draft (which 1) is common to both.
                    const bool is_main = (mode == DP_MODE_HYBRID_STREAM)
                        ? (sh.which == DP_STATE_TARGET_RECURRENT)
                        : (sh.which == DP_STATE_TARGET_WHOLE);
                    const bool is_dft = (sh.which == DP_STATE_DRAFT_WHOLE);
                    if (is_main) {
                        if (have_state_main) { fail("malformed state frame"); return false; }
                        state_main.assign(payload.begin() + sizeof(sh), payload.end());
                        have_state_main = true;
                    } else if (is_dft) {
                        if (have_state_dft) { fail("malformed state frame"); return false; }
                        state_dft.assign(payload.begin() + sizeof(sh), payload.end());
                        have_state_dft = true;
                    } else {
                        fail("unexpected state frame");
                        return false;
                    }
                    return true;
                } else if (cmd == DP_CMD_DONE) {
                    if (mode == DP_MODE_WHOLE) {
                        if (!have_state_main || (want_dft && !have_state_dft)) { fail("incomplete whole-mode state"); return false; }
                        finish_ok();
                        cb_whole(id_slot, id_task, gen, std::move(state_main), std::move(state_dft));
                        *done_flag = true;
                        return false;
                    }
                    if (mode == DP_MODE_HYBRID_STREAM) {
                        // attention CHUNKs already delivered via cb_chunk; deliver the
                        // recurrent tail (+ optional draft) for a single non-append apply.
                        if (!have_state_main || (want_dft && !have_state_dft)) { fail("incomplete hybrid-stream state"); return false; }
                        finish_ok();
                        cb_recur(id_slot, id_task, gen, std::move(state_main), std::move(state_dft));
                        *done_flag = true;
                        return false;
                    }
                    finish_ok();
                    cb_done(id_slot, id_task, gen, true, "");
                    *done_flag = true;
                    return false;
                } else if (cmd == DP_CMD_ERROR) {
                    dp_msg_error_hdr eh = {};
                    std::string msg = "remote error";
                    if (payload.size() >= sizeof(eh)) {
                        memcpy(&eh, payload.data(), sizeof(eh));
                        msg.assign((const char *) payload.data() + sizeof(eh),
                                   std::min((size_t) eh.msg_len, payload.size() - sizeof(eh)));
                    }
                    fail(msg);
                    return false;
                }
                fail("unexpected message");
                return false;
            };

            int         status   = 0;
            std::string err_body;
            bool        oversize = false;

            httplib::Request req;
            req.method = "POST";
            req.path   = "/v1/prefill";
            req.body   = body.dump();
            req.set_header("Content-Type", "application/json");
            if (!api_key_local.empty()) {
                req.set_header("Authorization", "Bearer " + api_key_local);
            }
            // the status must be known before the body is interpreted: a non-200
            // response is a JSON error body, not the frame stream.
            req.response_handler = [&](const httplib::Response & resp) {
                status = resp.status;
                return true;
            };
            req.content_receiver = [&](const char * data, size_t len, size_t, size_t) -> bool {
                if (status != 200) {
                    // cap buffering: the body only feeds an error log line, so a
                    // misbehaving peer must not be able to OOM us.
                    const size_t cap = 4096;
                    if (err_body.size() < cap) {
                        err_body.append(data, std::min(len, cap - err_body.size()));
                    }
                    return true;
                }
                if (!parser.feed(data, len)) {
                    oversize = true;
                    return false; // reject an oversized declared frame
                }
                return !*done_flag; // stop receiving once a terminal frame fired
            };

            httplib::Result res = cli.send(req);

            // a terminal frame already ran the bookkeeping and set *done_flag
            if (*done_flag) {
                return;
            }
            if (oversize) {
                fail("oversized frame from prefill peer");
                return;
            }
            if (status == 0) {
                // no response headers: transport failure, or a cancel() that
                // aborted send() before the peer replied.
                fail("prefill request failed: " + std::string(res ? "no response" : httplib::to_string(res.error())));
                return;
            }
            if (status != 200) {
                std::string msg = err_body;
                try {
                    json ej = json::parse(err_body);
                    if (ej.contains("error") && ej["error"].is_object() && ej["error"].contains("message")) {
                        msg = ej["error"]["message"].get<std::string>();
                    } else if (ej.contains("message")) {
                        msg = ej["message"].get<std::string>();
                    }
                } catch (const std::exception &) {}
                fail("prefill peer returned HTTP " + std::to_string(status) + ": " + msg);
                return;
            }
            // 200 but the stream ended without a DONE/ERROR terminal
            fail("stream interrupted");
        } catch (const std::exception & e) {
            LOG_WRN("prefill worker: exception: %s\n", e.what());
            if (!*done_flag) {
                fail("internal error");
            }
        }
    });

    workers.push_back({ std::move(th), done_flag });
    out_gen = gen;
    return true;
}

void server_prefill_client::cancel(int id_slot, bool user_initiated) {
    std::lock_guard<std::mutex> lk(mtx);
    auto it = clients_by_slot.find(id_slot);
    if (it == clients_by_slot.end()) {
        return;
    }
    if (user_initiated) {
        cancelled_slots.insert(id_slot); // suppresses the cooldown in fail()
    }
    // httplib::Client::stop() is thread-safe by design (shuts down the socket
    // underlying the in-flight Get()/send()); held under mtx together with the
    // worker's register/erase, so the entry cannot be dropped and cli destroyed
    // while stop() is touching it.
    it->second->stop();
}

void server_prefill_client::stop() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto & [id_slot, cli] : clients_by_slot) {
            cli->stop(); // see cancel(): cross-thread abort of a blocked worker
        }
    }
    for (auto & w : workers) {
        if (w.thread.joinable()) {
            w.thread.join();
        }
    }
    workers.clear();
}

bool dp_parse_host_port(const std::string & s, std::string & host, int & port) {
    const size_t pos = s.rfind(':');
    if (pos == std::string::npos || pos + 1 >= s.size()) {
        return false;
    }
    const std::string port_str = s.substr(pos + 1);
    for (char c : port_str) {
        if (!isdigit((unsigned char) c)) {
            return false;
        }
    }
    const long p = strtol(port_str.c_str(), nullptr, 10);
    if (p < 1 || p > 65535) {
        return false;
    }
    host = s.substr(0, pos);
    port = (int) p;
    return true;
}

// --- HTTP prefill protocol ---

static constexpr size_t DP_FRAME_HDR = 1 + sizeof(uint64_t); // cmd + payload_size

const char * dp_state_mode_str(dp_state_mode mode) {
    switch (mode) {
        case DP_MODE_STREAM:        return "stream";
        case DP_MODE_WHOLE:         return "whole";
        case DP_MODE_HYBRID_STREAM: return "hybrid_stream";
    }
    return "unknown";
}

bool dp_state_mode_parse(const std::string & s, dp_state_mode & mode) {
    if (s == "stream")        { mode = DP_MODE_STREAM;        return true; }
    if (s == "whole")         { mode = DP_MODE_WHOLE;         return true; }
    if (s == "hybrid_stream") { mode = DP_MODE_HYBRID_STREAM; return true; }
    return false;
}

dp_hello dp_make_hello(const llama_model * model, uint32_t n_ctx, ggml_type type_k, ggml_type type_v, uint64_t lora_hash) {
    dp_hello h;
    h.proto_version = DP_PROTO_VERSION;
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    h.arch      = arch;
    h.n_layer   = llama_model_n_layer(model);
    h.n_embd    = llama_model_n_embd(model);
    h.n_head_kv = llama_model_n_head_kv(model);
    h.n_ctx     = n_ctx;
    h.type_k    = (int32_t) type_k;
    h.type_v    = (int32_t) type_v;
    h.lora_hash = lora_hash;
    return h;
}

bool dp_hello_compatible(const dp_hello & remote, const dp_hello & local, std::string & reason) {
    if (remote.proto_version != local.proto_version) { reason = "protocol version mismatch"; return false; }
    if (remote.arch != local.arch)                   { reason = "architecture mismatch";     return false; }
    if (remote.n_layer != local.n_layer)             { reason = "n_layer mismatch";          return false; }
    if (remote.n_embd != local.n_embd)               { reason = "n_embd mismatch";           return false; }
    if (remote.n_head_kv != local.n_head_kv)         { reason = "n_head_kv mismatch";         return false; }
    if (remote.type_k != local.type_k)               { reason = "type_k mismatch";           return false; }
    if (remote.type_v != local.type_v)               { reason = "type_v mismatch";           return false; }
    if (remote.lora_hash != local.lora_hash)         { reason = "lora config differs"; } // warning only
    return true;
}

// hello fields live at the top level of the request body (flat contract;
// unknown keys are ignored by the serving side)
json dp_hello_to_json(const dp_hello & hello) {
    return json{
        { "proto",      hello.proto_version },
        { "arch",       hello.arch },
        { "n_layer",    hello.n_layer },
        { "n_embd",     hello.n_embd },
        { "n_head_kv",  hello.n_head_kv },
        { "n_ctx",      hello.n_ctx },
        { "type_k",     hello.type_k },
        { "type_v",     hello.type_v },
        { "state_mode", dp_state_mode_str(hello.state_mode) },
        { "has_dft",    hello.has_dft },
        { "lora_hash",  std::to_string(hello.lora_hash) }, // uint64 is not JSON-number-safe
    };
}

// json numbers are wider than uint32; reject out-of-range instead of
// truncating. accepts both signed- and unsigned-typed json integers, since
// json built in C++ types non-negative literals as signed.
static bool dp_get_uint32(const json & j, const char * key, uint32_t & dst, std::string & err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number_integer()) { err = std::string("missing or invalid field: ") + key; return false; }
    if (it->is_number_unsigned()) {
        const uint64_t v = it->get<uint64_t>();
        if (v > UINT32_MAX) { err = std::string("field out of range: ") + key; return false; }
        dst = (uint32_t) v;
    } else {
        const int64_t v = it->get<int64_t>();
        if (v < 0 || v > (int64_t) UINT32_MAX) { err = std::string("field out of range: ") + key; return false; }
        dst = (uint32_t) v;
    }
    return true;
}

static bool dp_hello_from_json(const json & j, dp_hello & out, std::string & err) {
    auto get_int = [&](const char * key, int32_t & dst) -> bool {
        auto it = j.find(key);
        if (it == j.end() || !it->is_number_integer()) { err = std::string("missing or invalid field: ") + key; return false; }
        if (it->is_number_unsigned()) {
            const uint64_t v = it->get<uint64_t>();
            if (v > INT32_MAX) { err = std::string("field out of range: ") + key; return false; }
            dst = (int32_t) v;
        } else {
            const int64_t v = it->get<int64_t>();
            if (v < INT32_MIN || v > INT32_MAX) { err = std::string("field out of range: ") + key; return false; }
            dst = (int32_t) v;
        }
        return true;
    };

    if (!dp_get_uint32(j, "proto", out.proto_version, err)) { return false; }

    auto it_arch = j.find("arch");
    if (it_arch == j.end() || !it_arch->is_string()) { err = "missing or invalid field: arch"; return false; }
    out.arch = it_arch->get<std::string>();

    if (!get_int("n_layer", out.n_layer))           { return false; }
    if (!get_int("n_embd", out.n_embd))             { return false; }
    if (!get_int("n_head_kv", out.n_head_kv))       { return false; }
    if (!dp_get_uint32(j, "n_ctx", out.n_ctx, err)) { return false; }
    if (!get_int("type_k", out.type_k))             { return false; }
    if (!get_int("type_v", out.type_v))             { return false; }

    auto it_mode = j.find("state_mode");
    if (it_mode == j.end() || !it_mode->is_string() || !dp_state_mode_parse(it_mode->get<std::string>(), out.state_mode)) {
        err = "missing or invalid field: state_mode";
        return false;
    }

    auto it_dft = j.find("has_dft");
    if (it_dft == j.end() || !it_dft->is_boolean()) { err = "missing or invalid field: has_dft"; return false; }
    out.has_dft = it_dft->get<bool>();

    auto it_lora = j.find("lora_hash");
    if (it_lora == j.end() || !it_lora->is_string()) { err = "missing or invalid field: lora_hash"; return false; }
    const std::string lora_str = it_lora->get<std::string>();
    if (lora_str.empty() || lora_str.find_first_not_of("0123456789") != std::string::npos) {
        err = "invalid field: lora_hash";
        return false;
    }
    out.lora_hash = strtoull(lora_str.c_str(), nullptr, 10);

    return true;
}

bool dp_prefill_request_parse(const json & body, dp_prefill_request & out, std::string & err) {
    if (!body.is_object()) { err = "request body must be a JSON object"; return false; }

    if (!dp_hello_from_json(body, out.client, err)) { return false; }
    if (out.client.proto_version != DP_PROTO_VERSION) { err = "protocol version mismatch"; return false; }
    out.state_mode = out.client.state_mode; // one flat state_mode key covers both

    auto it_tokens = body.find("tokens");
    if (it_tokens == body.end() || !it_tokens->is_array()) { err = "missing or invalid field: tokens"; return false; }
    if (it_tokens->empty()) { err = "tokens must be a non-empty array"; return false; }
    out.tokens.clear();
    out.tokens.reserve(it_tokens->size());
    for (const auto & t : *it_tokens) {
        if (!t.is_number_integer()) { err = "tokens must be an array of integers"; return false; }
        if (t.is_number_unsigned()) {
            if (t.get<uint64_t>() > INT32_MAX) { err = "token value out of range"; return false; }
        } else {
            const int64_t v = t.get<int64_t>();
            if (v < INT32_MIN || v > INT32_MAX) { err = "token value out of range"; return false; }
        }
        out.tokens.push_back(t.get<llama_token>());
    }

    if (!dp_get_uint32(body, "p0", out.p0, err)) { return false; }
    if (out.p0 >= out.tokens.size()) { err = "p0 out of range"; return false; }

    auto it_dft = body.find("want_dft");
    if (it_dft != body.end()) {
        if (!it_dft->is_boolean()) { err = "want_dft must be a boolean"; return false; }
        out.want_dft = it_dft->get<bool>();
    } else {
        out.want_dft = false;
    }

    return true;
}

// --- dp_pipe ---

// dp_frame(), but built directly into the std::string dp_pipe queues
static std::string dp_frame_str(dp_cmd cmd, const void * hdr, size_t hdr_size,
                                const uint8_t * body = nullptr, size_t body_size = 0) {
    const uint64_t payload_size = hdr_size + body_size;
    std::string f(1 + sizeof(payload_size) + payload_size, '\0');
    char * p = &f[0];
    *p++ = (char) cmd;
    memcpy(p, &payload_size, sizeof(payload_size)); p += sizeof(payload_size);
    memcpy(p, hdr, hdr_size);                       p += hdr_size;
    if (body_size > 0) {
        memcpy(p, body, body_size);
    }
    return f;
}

void dp_pipe::enqueue(std::string && frame, bool is_terminal) {
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (reader_closed) {
            return; // sends are no-ops once the reader has gone away
        }
        queue.push_back(std::move(frame));
        if (is_terminal) {
            terminal = true;
        }
    }
    cv.notify_one();
}

void dp_pipe::send_chunk(uint32_t p0, uint32_t p1, std::vector<uint8_t> && blob) {
    dp_msg_chunk_hdr hdr = { p0, p1, (uint64_t) blob.size() };
    enqueue(dp_frame_str(DP_CMD_CHUNK, &hdr, sizeof(hdr), blob.data(), blob.size()));
}

void dp_pipe::send_progress(uint32_t n_processed) {
    dp_msg_progress msg = { n_processed };
    enqueue(dp_frame_str(DP_CMD_PROGRESS, &msg, sizeof(msg)));
}

void dp_pipe::send_state(uint8_t which, std::vector<uint8_t> && blob) {
    dp_msg_state_hdr hdr = { which, (uint64_t) blob.size() };
    enqueue(dp_frame_str(DP_CMD_STATE, &hdr, sizeof(hdr), blob.data(), blob.size()));
}

void dp_pipe::send_done(uint32_t n_tokens_total) {
    dp_msg_done msg = { n_tokens_total };
    enqueue(dp_frame_str(DP_CMD_DONE, &msg, sizeof(msg)), /*is_terminal=*/true);
}

void dp_pipe::send_error(int32_t code, const std::string & msg) {
    dp_msg_error_hdr hdr = { code, (uint32_t) msg.size() };
    enqueue(dp_frame_str(DP_CMD_ERROR, &hdr, sizeof(hdr), (const uint8_t *) msg.data(), msg.size()), /*is_terminal=*/true);
}

bool dp_pipe::alive() const {
    std::lock_guard<std::mutex> lk(mtx);
    return !reader_closed;
}

bool dp_pipe::finished() const {
    std::lock_guard<std::mutex> lk(mtx);
    return terminal;
}

bool dp_pipe::read(std::string & out, const std::function<bool()> & should_stop) {
    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        if (!queue.empty()) {
            out = std::move(queue.front());
            queue.pop_front();
            return true;
        }
        // drained: once the terminal record has been handed out there is
        // nothing more coming, so signal a clean end
        if (terminal || reader_closed) {
            return false;
        }
        // evaluate should_stop() outside the lock so a callback that re-enters
        // the pipe (e.g. alive()) cannot self-deadlock
        lk.unlock();
        const bool stop = should_stop && should_stop();
        lk.lock();
        if (stop) {
            return false;
        }
        if (queue.empty() && !terminal && !reader_closed) {
            cv.wait_for(lk, std::chrono::milliseconds(500));
        }
    }
}

void dp_pipe::close_read() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        reader_closed = true;
    }
    cv.notify_all(); // wake a reader blocked in read()
}

// --- dp_frame_parser ---

bool dp_frame_parser::feed(const char * data, size_t len) {
    buf.insert(buf.end(), (const uint8_t *) data, (const uint8_t *) data + len);

    size_t pos = 0;
    while (true) {
        if (!have_header) {
            if (buf.size() - pos < DP_FRAME_HDR) {
                break;
            }
            cur_cmd = (dp_cmd) buf[pos];
            uint64_t sz;
            memcpy(&sz, buf.data() + pos + 1, sizeof(sz));
            if (sz > DP_MAX_PAYLOAD) {
                return false; // reject before allocating
            }
            cur_size = (size_t) sz;
            pos += DP_FRAME_HDR;
            have_header = true;
        }
        if (buf.size() - pos < cur_size) {
            break;
        }
        std::vector<uint8_t> payload(buf.begin() + pos, buf.begin() + pos + cur_size);
        pos += cur_size;
        have_header = false;
        if (on_record && !on_record(cur_cmd, std::move(payload))) {
            buf.erase(buf.begin(), buf.begin() + pos);
            return true; // consumer asked to stop
        }
    }
    buf.erase(buf.begin(), buf.begin() + pos);
    return true;
}
