#include "MAC.hpp"
#include "ComputeTiming.hpp"

// Numerical execution and cache updates only; the MAC state machine schedules
// completion and network injection in MAC.cpp.
float MAC::computeTransformerElement() {
    float result = 0.0f;
    if (fn == LAYERNORM) {                                                         // LayerNorm
        // infeature = [mean, var, x_i, gamma, beta]
        float mean  = infeature[0];
        float var   = infeature[1];
        float x_i   = infeature[2];
        float gamma = infeature[3];
        float beta  = infeature[4];
        result = (x_i - mean) / std::sqrt(var + 1e-5) * gamma + beta;
    }
    else if (fn == SOFTMAX_TR) {                                                        // Softmax
        assert(m_size > 0 && "SOFTMAX_TR: m_size cannot be 0, check inbuffer after request");
        assert((int)infeature.size() >= m_size && "SOFTMAX_TR: infeature too short for m_size");

        int target_idx = tmpm % m_size;
        int seq_idx = tmpm / m_size;
        if (causal_mask == 1 && target_idx > seq_idx) {
            result = 0.0; // causal masking
        } else {
            float max_val = -1e9;
            int limit = (causal_mask == 1) ? seq_idx : m_size - 1;
            for(int j=0; j <= limit; j++) { if (infeature[j] > max_val) max_val = infeature[j]; }
            float sum_exp = 0.0;
            for(int j=0; j <= limit; j++) { sum_exp += std::exp(infeature[j] - max_val); }
            result = std::exp(infeature[target_idx] - max_val) / sum_exp;
        }
    }
    else if (fn == ADD) {                                                               // Add (Residual Connection)
        result = infeature[0] + infeature[1];
    }
    else if (fn == EMBEDDING) {                                                         // Embedding Lookup
        int idx = tmpm % m_size;
        result = weight[idx] * std::sqrt(static_cast<float>(m_size));
    }
    else if (fn == RMSNORM) {                                                           // RMSNorm
        // infeature contains [rms, x_i, gamma]
        float rms   = infeature[0];
        float x_i   = infeature[1];
        float gamma = infeature[2];
        result = (x_i / rms) * gamma;
    }
    else if (fn == SWIGLU) {                                                            // SwiGLU
        float gate = infeature[0];
        float up   = infeature[1];

        float silu = gate * (1.0 / (1.0 + std::exp(-gate)));
        result = silu * up;
    }
    else if (fn == GEGLU) {
        float gate = infeature[0];
        float up   = infeature[1];
        float gelu = 0.5f * gate * (1.0f + std::erf(gate / 1.41421356f));
        result = gelu * up;
    }
    else if (fn == ROPE) {                                                              // RoPE
        int pos = inbuffer[2];
        int head_dim = inbuffer[3];
        int idx = tmpm % m_size;

        int d = idx % head_dim;
        int half_dim = head_dim / 2;
        int feat_idx = (d < half_dim) ? d : (d - half_dim);

        float freq = 1.0 / std::pow(10000.0, ((float)(feat_idx * 2) / head_dim));
        float theta = pos * freq;
        float cos_val = std::cos(theta);
        float sin_val = std::sin(theta);

        float val_curr = infeature[0];
        float val_pair = infeature[1];

        if (d < half_dim) {
            result = val_curr * cos_val - val_pair * sin_val;
        } else {
            result = val_curr * cos_val + val_pair * sin_val;
        }
    }
    return result;
}

PEAttentionWork MAC::computeAttentionElement() {
    // Header: op, q_dim, k_dim, n_heads, row, target_idx,
    //         kv_head_id, kv_start_token, kv_token_count.
    // Payload: query head, followed by [K head, V head] per token.
    int q_dim = inbuffer[1];
    int k_dim = inbuffer[2];
    int n_heads = inbuffer[3];
    int current_row = inbuffer[4];
    int target_idx = inbuffer[5];
    int kv_head_id = inbuffer[6];
    int kv_start_token = inbuffer[7];
    int kv_token_count = inbuffer[8];

    int head_dim = q_dim / n_heads;
    int k_head_dim = head_dim;
    int my_head = target_idx / head_dim;
    int target_d = target_idx % head_dim;
    int kv_size = k_head_dim * 2;
    int k_half_dim = k_head_dim / 2;
    int half_dim = head_dim / 2;
    assert(kv_head_id == my_head * (k_dim / k_head_dim) / n_heads);

    if (cached_layer_id != net->c_layer) {
        cached_score_row = -1;
        cached_score_head = -1;
    }
    if (cached_layer_id != net->c_layer || cached_kv_head_id != kv_head_id) {
        kv_cache.clear();
        cached_through_token = -1;
        cached_layer_id = net->c_layer;
        cached_kv_head_id = kv_head_id;
    }

    int expected_cache_size = (current_row + 1) * kv_size;
    assert(expected_cache_size <= KV_CACHE_SIZE && "Head-local KV cache overflow");
    if (kv_token_count > 0) {
        if (kv_start_token == 0) {
            kv_cache.clear();
            cached_through_token = -1;
        }
        assert(kv_start_token == cached_through_token + 1);
        for (int t = kv_start_token; t < kv_start_token + kv_token_count; t++) {
            int t_offset = 9 + head_dim + (t - kv_start_token) * kv_size;
            std::vector<float> token_k_rotated(k_head_dim, 0.0);
            for (int d = 0; d < k_half_dim; d++) {
                float freq = 1.0 / std::pow(10000.0, (float)(2 * d) / k_head_dim);
                float theta = t * freq;
                float cos_val = std::cos(theta);
                float sin_val = std::sin(theta);

                float k1 = inbuffer[t_offset + d];
                float k2 = inbuffer[t_offset + d + k_half_dim];
                token_k_rotated[d] = k1 * cos_val - k2 * sin_val;
                token_k_rotated[d + k_half_dim] = k2 * cos_val + k1 * sin_val;
            }
            kv_cache.insert(kv_cache.end(), token_k_rotated.begin(), token_k_rotated.end());
            kv_cache.insert(kv_cache.end(), inbuffer.begin() + t_offset + k_head_dim,
                            inbuffer.begin() + t_offset + kv_size);
        }
        cached_through_token = kv_start_token + kv_token_count - 1;
    }
    recordSramUsage();
    assert(kv_cache.size() == expected_cache_size && "Head-local KV cache size mismatch");

    // calculating score (O(N)) or cache recovery (O(1)) ---
    int SINK_TOKENS = 4; // Deve coincidere con il numero di Sinks del router
    std::vector<int> valid_tokens;

    for (int t = 0; t <= current_row; t++) {
        if (current_row >= MAX_CONTEXT_WINDOW) {
            int max_recent_tokens = MAX_CONTEXT_WINDOW - SINK_TOKENS;
            // Salta i token intermedi (simula l'eviction del Ring Buffer nel router)
            if (t >= SINK_TOKENS && t < current_row - max_recent_tokens + 1) {
                continue;
            }
        }
        valid_tokens.push_back(t);
    }
    int effective_history = valid_tokens.size();

    // Checking the local cache (based on the window, not the entire history)
    bool scores_hit = (this->cached_score_row == current_row && this->cached_score_head == my_head);
    std::vector<float> final_scores(effective_history, 0.0);

    const bool rebuild_scores = !scores_hit || this->cached_attention_scores.size() != effective_history;
    if (rebuild_scores) {
        // RoPE for the Query
        std::vector<float> q_rotated(head_dim, 0.0);

        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0 / std::pow(10000.0, (float)(2 * d) / head_dim);
            float theta = current_row * freq;
            float cos_val = std::cos(theta);
            float sin_val = std::sin(theta);

            float q1 = infeature[d];
            float q2 = infeature[d + half_dim];

            q_rotated[d]            = q1 * cos_val - q2 * sin_val;
            q_rotated[d + half_dim] = q2 * cos_val + q1 * sin_val;
        }

        // Dot Product (Q * K^T) for the current window
        float max_score = -1e9;

        for (int i = 0; i < effective_history; i++) {
            int t = valid_tokens[i];

            float dot_product = 0.0;
            int k_start = t * kv_size;

            for (int d = 0; d < k_head_dim; d++) {
                dot_product += q_rotated[d] * this->kv_cache[k_start + d];
            }

            final_scores[i] = dot_product / std::sqrt((float)head_dim);

            if (final_scores[i] > max_score) {
                max_score = final_scores[i];
            }
        }

        // Softmax
        float sum_exp = 0.0;
        for (int i = 0; i < effective_history; i++) {
            final_scores[i] = std::exp(final_scores[i] - max_score);
            sum_exp += final_scores[i];
        }
        for (int i = 0; i < effective_history; i++) {
            final_scores[i] /= sum_exp;
        }

        // cache score update
        this->cached_attention_scores = final_scores;
        recordSramUsage();
        this->cached_score_row = current_row;
        this->cached_score_head = my_head;

    } else {
        // cache hit, reuse the cache scores
        final_scores = this->cached_attention_scores;
    }

    // value projection
    outfeature = 0.0;
    for (int i = 0; i < effective_history; i++) {
        int t = valid_tokens[i];
        int v_start = (t * kv_size) + k_head_dim;
        outfeature += final_scores[i] * this->kv_cache[v_start + target_d];
    }

    this->tmpm = (current_row * q_dim) + target_idx;
    return {effective_history, head_dim, rebuild_scores};
}

void MAC::sigmoid(float& x) // 3
{
    x = 1.0 / (1.0 + std::exp(-x));
}

void MAC::tanh(float& x)  // 2
{
    x = 2.0 / (1.0 + std::exp(-2 * x)) - 1;
}

void MAC::relu(float& x)  // 1
{
    if (x < 0) x = 0.0;
}
