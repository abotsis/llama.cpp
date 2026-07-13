#include "llama.h"
#include "common.h"
#include "get-model.h"

#include <cstdio>
#include <cstring>
#include <vector>

// verifies that a sequence state transferred as position-range chunks with
// LLAMA_STATE_SEQ_FLAGS_APPEND is bitwise-equivalent to a whole-blob transfer.
//
// For a plain hybrid model (recurrent/SSM layers + full-attention layers) the
// range chunks only carry the attention KV; the bounded recurrent state is sent
// once as a whole blob via LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY. The split restore
// (attention ranges + recurrent whole) must produce logits that are bitwise
// identical to a legacy whole-blob restore. On a dense model there is no
// recurrent part, so the attention ranges alone reconstruct the whole state.
//
// Memory types that do not support ranges (pure recurrent, SWA/iswa, and
// hybrid-iswa) must report 0 for every bounded range; the test detects this via
// the public API and instead checks the legacy whole round-trip for them.
//
// ctest runs this twice: once against the in-repo dense model, and once as
// test-state-seq-range-hybrid against a small hybrid model (LFM2-test-ci-80M).
// To run by hand against another model:
//     LLAMACPP_TEST_MODELFILE=/path/to/model.gguf ./bin/test-state-seq-range
//   or: ./bin/test-state-seq-range /path/to/model.gguf

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 256;
    cparams.n_seq_max = 2; // need a second sequence to prove APPEND isolation
    return llama_init_from_model(model, cparams);
}

static void decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int p0, bool want_logits, llama_seq_id seq_id = 0) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], p0 + (int) i, { seq_id }, false);
    }
    if (want_logits) {
        batch.logits[batch.n_tokens - 1] = true;
    }
    const int ret = llama_decode(ctx, batch);
    GGML_ASSERT(ret == 0);
    llama_batch_free(batch);
}

// whole per-sequence state blob (attention + recurrent), used as a reference
static std::vector<uint8_t> snapshot_seq(llama_context * ctx, llama_seq_id seq_id) {
    const size_t n = llama_state_seq_get_size_ext(ctx, seq_id, 0);
    std::vector<uint8_t> buf(n);
    if (n > 0) {
        const size_t got = llama_state_seq_get_data_ext(ctx, buf.data(), n, seq_id, 0);
        GGML_ASSERT(got == n);
    }
    return buf;
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // keep it deterministic and cheap
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    GGML_ASSERT(model);

    const bool is_hybrid = llama_model_is_hybrid(model);
    printf("test-state-seq-range: model is %s\n", is_hybrid ? "hybrid" : "dense/non-hybrid");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // build a prompt of exactly 96 valid tokens
    std::vector<llama_token> tokens;
    {
        const char * text = "Once upon a time there was a little dog named Rex. "
                            "Rex liked to run and play in the park every day. ";
        std::string prompt;
        while ((int) tokens.size() < 96) {
            prompt += text;
            tokens.resize(256);
            const int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                         tokens.data(), (int32_t) tokens.size(), true, false);
            GGML_ASSERT(n > 0);
            tokens.resize(n);
        }
        tokens.resize(96);
    }

    llama_context * ctx_a = make_ctx(model); // source: does the prefill
    llama_context * ctx_b = make_ctx(model); // dest: split (or whole) restore
    llama_context * ctx_c = make_ctx(model); // dest: legacy whole-blob restore

    // (a) prefill the source sequence
    decode_tokens(ctx_a, tokens, 0, false);

    // whole blob from ctx_a (legacy path: attention + recurrent, byte-identical to before)
    std::vector<uint8_t> blob_full(llama_state_seq_get_size_ext(ctx_a, 0, 0));
    GGML_ASSERT(blob_full.size() > 0);
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx_a, blob_full.data(), blob_full.size(), 0, 0) == blob_full.size());

    // detect whether this memory type supports position ranges. plain hybrid and
    // dense report a non-zero size for a bounded range; pure recurrent, iswa and
    // hybrid-iswa report 0 (ranges are a non-goal for them).
    const llama_pos bounds[4] = { 0, 32, 64, 96 };
    const bool range_supported = llama_state_seq_get_size_range(ctx_a, 0, bounds[0], bounds[1], 0) > 0;
    printf("test-state-seq-range: position ranges %s\n", range_supported ? "supported" : "not supported (whole-mode only)");

    // populate a second, unrelated sequence (seq 1) in ctx_b BEFORE restoring seq 0,
    // so we can prove the seq-0 restore leaves it untouched
    const std::vector<llama_token> other(tokens.begin(), tokens.begin() + 32);
    decode_tokens(ctx_b, other, 0, false, /*seq_id*/ 1);
    const std::vector<uint8_t> seq1_before = snapshot_seq(ctx_b, 1);
    GGML_ASSERT(seq1_before.size() > 0);

    // (b) legacy whole-blob restore into ctx_c seq 0 (the reference path)
    GGML_ASSERT(llama_state_seq_set_data_ext(ctx_c, blob_full.data(), blob_full.size(), 0, 0) == blob_full.size());

    if (range_supported) {
        // (c) split restore into ctx_b seq 0:
        // attention sub-state as position-range chunks [0,32) [32,64) [64,96),
        // each applied with APPEND in ascending position order
        for (int i = 0; i < 3; ++i) {
            const llama_pos p0 = bounds[i];
            const llama_pos p1 = bounds[i + 1];

            const size_t size = llama_state_seq_get_size_range(ctx_a, 0, p0, p1, 0);
            GGML_ASSERT(size > 0);

            std::vector<uint8_t> chunk(size);
            GGML_ASSERT(llama_state_seq_get_data_range(ctx_a, chunk.data(), size, 0, p0, p1, 0) == size);

            // append into ctx_b (first chunk appends into an empty seq)
            GGML_ASSERT(llama_state_seq_set_data_ext(ctx_b, chunk.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_APPEND) == size);
        }

        // on a hybrid model the ranges above carried only the attention KV; transfer
        // the bounded recurrent sub-state once, whole, via PARTIAL_ONLY
        if (is_hybrid) {
            const size_t recr_size = llama_state_seq_get_size_ext(ctx_a, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            GGML_ASSERT(recr_size > 0);
            std::vector<uint8_t> recr(recr_size);
            GGML_ASSERT(llama_state_seq_get_data_ext(ctx_a, recr.data(), recr_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == recr_size);
            GGML_ASSERT(llama_state_seq_set_data_ext(ctx_b, recr.data(), recr_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == recr_size);
        }

        // an unbounded range call must equal the full blob size (whole state, both sub-parts)
        GGML_ASSERT(llama_state_seq_get_size_range(ctx_a, 0, -1, -1, 0) == blob_full.size());
    } else {
        // non-goal guard: every bounded range must report 0 for this memory type
        for (int i = 0; i < 3; ++i) {
            GGML_ASSERT(llama_state_seq_get_size_range(ctx_a, 0, bounds[i], bounds[i + 1], 0) == 0);
        }
        // reconstruct seq 0 in ctx_b via the legacy whole blob instead
        GGML_ASSERT(llama_state_seq_set_data_ext(ctx_b, blob_full.data(), blob_full.size(), 0, 0) == blob_full.size());
    }

    // the seq-0 restore must not have disturbed the untouched second sequence
    const std::vector<uint8_t> seq1_after = snapshot_seq(ctx_b, 1);
    GGML_ASSERT(seq1_after.size() == seq1_before.size());
    GGML_ASSERT(memcmp(seq1_after.data(), seq1_before.data(), seq1_before.size()) == 0);

    // (d) decode one identical token at pos 96 on both dests and compare logits bitwise
    const std::vector<llama_token> probe = { tokens.back() };
    decode_tokens(ctx_b, probe, 96, true);
    decode_tokens(ctx_c, probe, 96, true);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits_b = llama_get_logits_ith(ctx_b, -1);
    const float * logits_c = llama_get_logits_ith(ctx_c, -1);
    GGML_ASSERT(logits_b && logits_c);
    GGML_ASSERT(memcmp(logits_b, logits_c, sizeof(float) * n_vocab) == 0);

    printf("test-state-seq-range: OK\n");

    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_free(ctx_c);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
