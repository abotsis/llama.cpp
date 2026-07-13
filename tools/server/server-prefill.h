#pragma once

// disaggregated prefill: a small protocol that lets one llama-server delegate
// prompt processing to another and get the resulting KV state back, either as
// streamed position-range chunks (stream mode) or as one whole-context blob
// per sequence (whole mode, for SWA/hybrid/recurrent models). framing follows
// the ggml-rpc style:
//   uint8_t cmd | uint64_t payload_size (LE) | payload

#include "llama.h"

#include <nlohmann/json.hpp>

namespace httplib { class Client; }

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

#define DP_PROTO_VERSION 3

// framing cap shared by both sides: dp_frame_parser::feed() enforces it on the
// delegating side, and the whole-mode state serializer (server-context.cpp)
// checks a blob against it before attempting to send, so an oversized state
// fails loudly instead of tripping the generic frame-size guard on the wire.
static constexpr size_t DP_MAX_PAYLOAD = 16ull * 1024 * 1024 * 1024; // 16 GiB
// ^ sized to cover a full-context whole-mode blob for large hybrid models: the
// per-sequence state grows ~linearly with tokens (~55 KB/tok measured on a 35B
// hybrid), so a 128k-token prompt serializes to ~7 GiB, plus a separate MTP
// draft frame. Frame headers are uint64 and every size on the wire/parser path
// is size_t, so the only real cost of a large blob is transient host memory.

// streamed-body frame commands (uint8 cmd | uint64 payload_size | payload);
// values are wire-stable
enum dp_cmd : uint8_t {
    DP_CMD_CHUNK    = 2,
    DP_CMD_DONE     = 3,
    DP_CMD_ERROR    = 4,
    DP_CMD_PROGRESS = 5, // whole mode only: keeps the stream timeout from firing
    DP_CMD_STATE    = 6, // whole & hybrid-stream: a whole-context or recurrent state blob
};

// how the serve side hands KV state back: STREAM = v1 position-range chunks
// (dense models only), WHOLE = one whole-sequence state blob (works for any
// model, including SWA/hybrid/recurrent), HYBRID_STREAM = plain (non-SWA) hybrid
// models: attention KV streamed by position range + recurrent state sent whole.
// values are wire-stable - append only.
enum dp_state_mode : uint8_t {
    DP_MODE_STREAM        = 0,
    DP_MODE_WHOLE         = 1,
    DP_MODE_HYBRID_STREAM = 2,
};

#pragma pack(push, 1)
struct dp_msg_chunk_hdr {
    uint32_t p0;
    uint32_t p1;
    uint64_t n_bytes;     // size of the state blob that follows
};

struct dp_msg_done {
    uint32_t n_tokens_total;
};

struct dp_msg_error_hdr {
    int32_t  code;
    uint32_t msg_len;     // followed by msg_len bytes (not nul-terminated)
};

struct dp_msg_progress {
    uint32_t n_processed;
};

struct dp_msg_state_hdr {
    uint8_t  which;       // see dp_state_which
    uint64_t n_bytes;     // size of the state blob that follows
};
#pragma pack(pop)

// dp_msg_state_hdr::which - identifies which serialized state a STATE frame
// carries. wire-stable, append only.
enum dp_state_which : uint8_t {
    DP_STATE_TARGET_WHOLE     = 0, // whole mode: full target sequence state
    DP_STATE_DRAFT_WHOLE      = 1, // whole draft (MTP) context state
    DP_STATE_TARGET_RECURRENT = 2, // hybrid-stream: target recurrent state (PARTIAL_ONLY)
};

// error codes
enum dp_error_code : int32_t {
    DP_ERR_PROTO      = 1,
    DP_ERR_TOO_LONG   = 2,
    DP_ERR_INTERNAL   = 3,
};

bool dp_parse_host_port(const std::string & s, std::string & host, int & port);

// --- HTTP prefill protocol ---
// the KV state is streamed back over HTTP (POST /v1/prefill); the framing
// (uint8 cmd | uint64 payload_size | payload) is kept for the streamed body.

const char * dp_state_mode_str(dp_state_mode mode);
bool         dp_state_mode_parse(const std::string & s, dp_state_mode & mode);

// hybrid-stream shares whole mode's DELEGATION-LAUNCH decisions (delegate the
// full token list minus one, p0=0, request the draft whole) because the
// recurrent state cannot be rewound to a cached prefix. The RECEIVE/APPLY path
// differs: hybrid-stream streams the attention KV by range (APPEND chunks) and
// restores only the recurrent tail whole (PARTIAL_ONLY), whereas whole mode
// replaces the entire sequence. This helper gates the shared launch decisions
// only; apply sites branch on the mode explicitly.
inline bool dp_mode_transfers_whole(dp_state_mode mode) {
    return mode == DP_MODE_WHOLE || mode == DP_MODE_HYBRID_STREAM;
}

// HELLO descriptor exchanged as flat JSON on POST /v1/prefill
struct dp_hello {
    uint32_t      proto_version = DP_PROTO_VERSION;
    std::string   arch;
    int32_t       n_layer = 0, n_embd = 0, n_head_kv = 0;
    uint32_t      n_ctx = 0;
    int32_t       type_k = 0, type_v = 0;
    dp_state_mode state_mode = DP_MODE_STREAM;
    bool          has_dft = false;
    uint64_t      lora_hash = 0;
};

dp_hello     dp_make_hello(const llama_model * model, uint32_t n_ctx, ggml_type type_k, ggml_type type_v, uint64_t lora_hash);
bool         dp_hello_compatible(const dp_hello & remote, const dp_hello & local, std::string & reason);
json         dp_hello_to_json(const dp_hello & hello);

// validated server-side view of a decoded prefill request body
struct dp_prefill_request {
    dp_hello                 client;
    std::vector<llama_token> tokens;
    uint32_t                 p0 = 0;
    dp_state_mode            state_mode = DP_MODE_STREAM;
    bool                     want_dft = false;
};
bool dp_prefill_request_parse(const json & body, dp_prefill_request & out, std::string & err);

// serving-side task carrier: the inference/queue thread pushes pre-framed
// records (non-blocking, unbounded) and the HTTP worker draining POST /v1/prefill
// pulls them via read(). sends become no-ops after close_read().
struct dp_pipe {
    int id_task = -1;

    // producer API (inference/queue thread; non-blocking)
    void send_chunk(uint32_t p0, uint32_t p1, std::vector<uint8_t> && blob);
    void send_progress(uint32_t n_processed);
    void send_state(uint8_t which, std::vector<uint8_t> && blob);
    void send_done(uint32_t n_tokens_total);
    void send_error(int32_t code, const std::string & msg);
    bool alive() const;    // false once close_read() has run
    bool finished() const; // true once DONE/ERROR has been enqueued

    // consumer API (HTTP worker draining into the response body)
    bool read(std::string & out, const std::function<bool()> & should_stop); // blocks, 500ms poll
    void close_read();

private:
    void enqueue(std::string && frame, bool is_terminal = false);

    mutable std::mutex      mtx;
    std::condition_variable cv;
    std::deque<std::string> queue;
    bool                    terminal = false;      // DONE/ERROR enqueued
    bool                    reader_closed = false; // close_read() called
};

// incremental client-side frame parser: accumulates a 9-byte header then its
// payload, calling on_record for each complete frame. on_record returns false
// to stop consuming further records; feed() returns false only on a declared
// payload_size exceeding DP_MAX_PAYLOAD (checked before allocating).
struct dp_frame_parser {
    bool feed(const char * data, size_t len);
    std::function<bool(dp_cmd, std::vector<uint8_t> &&)> on_record;

private:
    std::vector<uint8_t> buf;              // unconsumed bytes
    bool                 have_header = false;
    dp_cmd               cur_cmd = DP_CMD_CHUNK;
    size_t               cur_size = 0;     // payload size of the frame in progress
};

// delegating side: hands a slot's uncached prompt suffix to a remote peer and
// streams the resulting KV chunks back via callbacks. each delegation runs on
// its own worker thread; callers must only touch sockets/state under mtx.
struct server_prefill_client {
    using chunk_cb = std::function<void(int, int, uint64_t, uint32_t, uint32_t, std::vector<uint8_t> &&)>;
    using done_cb  = std::function<void(int, int, uint64_t, bool, const std::string &)>;
    // whole mode success only: id_slot, id_task, gen, main state blob, draft
    // (MTP) state blob (empty if not requested). called instead of on_done.
    using whole_cb = std::function<void(int, int, uint64_t, std::vector<uint8_t> &&, std::vector<uint8_t> &&)>;
    // hybrid-stream success only: same shape as whole_cb but the first blob is
    // the recurrent tail (applied PARTIAL_ONLY, not replacing the attention KV
    // the CHUNKs built); the second is the draft (MTP) state. called instead of
    // on_done after all attention CHUNKs have been delivered via on_chunk.
    using recurrent_cb = whole_cb;

    bool init(const std::string & host, int port, const std::string & api_key,
              const dp_hello & local_hello, int max_inflight,
              chunk_cb on_chunk, done_cb on_done, whole_cb on_whole, recurrent_cb on_recurrent);

    // starts a worker thread that POSTs the request (carrying mode/want_dft) to
    // the peer's /v1/prefill and pumps chunks into on_chunk (stream mode) or
    // collects state blobs and calls on_whole (whole mode); returns false if
    // saturated or cooling down. on success, out_gen receives a per-attempt
    // generation id (unique across all delegations for this client) that the
    // caller must echo back so stale messages from a superseded worker can be
    // told apart from the current attempt.
    bool try_delegate(int id_slot, int id_task, std::vector<llama_token> && tokens, uint32_t p0,
                       dp_state_mode mode, bool want_dft, uint64_t & out_gen);

    // abort that job's HTTP request; its worker exits via on_done(ok=false).
    // only user-initiated cancels (the default) suppress the 60s cooldown;
    // internal callers detecting mid-stream corruption pass false so the peer
    // is still penalized.
    void cancel(int id_slot, bool user_initiated = true);
    void stop();

    ~server_prefill_client() { stop(); }

private:
    void reap_finished();

    std::string   host;
    int           port = 0;
    std::string   api_key; // Bearer sent on /v1/prefill
    dp_hello      hello_local{};
    int           max_inflight = 1;
    chunk_cb      on_chunk;
    done_cb       on_done;
    whole_cb      on_whole;
    recurrent_cb  on_recurrent;

    std::mutex mtx;
    // id_slot -> the worker's own httplib::Client (stack-local, one per
    // delegation), so cancel()/stop() can call stop() on it from another thread
    // while the worker blocks in the streaming send(). the worker
    // registers/erases the entry under mtx (erase-only-if-still-mine), so
    // the pointer is only ever valid while the map holds it.
    std::map<int, httplib::Client *> clients_by_slot;
    std::set<int> cancelled_slots;
    int n_inflight = 0;
    int64_t t_cooldown_until_us = 0;
    uint64_t generation_next = 1; // next per-attempt id handed out by try_delegate()

    // a worker sets *done = true just before it returns; reap_finished() joins
    // and drops entries once they are marked done (never joins a live thread)
    struct worker_slot {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::vector<worker_slot> workers;
};
