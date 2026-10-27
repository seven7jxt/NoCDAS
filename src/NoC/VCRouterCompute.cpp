#include "VCRouter.hpp"
#include <cmath>
#include <algorithm>

std::vector<int> VCRouter::getAttentionPhysicalSlots(Flit* t_flit,
                                                     int router_idx,
                                                     int path_length,
                                                     int k_dim) const {
    std::vector<int> valid_physical_slots;
    if (router_idx < 0 || path_length <= 0 || k_dim <= 0) {
        return valid_physical_slots;
    }

    const int SINK_TOKENS = 4;
    const int floats_per_token = k_dim * 2;
    const int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;
    if (max_tokens_in_sram <= 0) {
        return valid_physical_slots;
    }

    const int current_query_y = t_flit->packet->message.sequence_id;
    int current_local_count = 0;
    for (int y = 0; y <= current_query_y; y++) {
        if (y % path_length == router_idx) current_local_count++;
    }

    int local_seq_id = 0;
    for (int y = 0; y <= current_query_y; y++) {
        if (y % path_length != router_idx) continue;

        bool retained = true;
        if (current_local_count > max_tokens_in_sram) {
            const int max_recent = max_tokens_in_sram - SINK_TOKENS;
            if (local_seq_id >= SINK_TOKENS &&
                local_seq_id < current_local_count - max_recent) {
                retained = false;
            }
        }

        if (retained) {
            int physical_slot;
            if (local_seq_id < max_tokens_in_sram) {
                physical_slot = local_seq_id;
            } else if (max_tokens_in_sram > SINK_TOKENS) {
                physical_slot = SINK_TOKENS +
                    ((local_seq_id - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
            } else {
                physical_slot = std::max(0, max_tokens_in_sram - 1);
            }
            valid_physical_slots.push_back(physical_slot);
        }
        local_seq_id++;
    }

    return valid_physical_slots;
}

int VCRouter::computeInTransit(Flit* t_flit, int port_idx) {
    int this_router_id = id[0] * X_NUM + id[1];
    int matmul_macs = 0;

    // Avoid redundant executions of the same flit if it passes multiple times
    // if (std::find(t_flit->computed_routers.begin(), t_flit->computed_routers.end(), this_router_id) != t_flit->computed_routers.end()) {
    //     return;
    // }
    // t_flit->computed_routers.push_back(this_router_id);

    // Avoid redundant executions of the same flit if it passes multiple times
    if (t_flit->computed_routers[this_router_id]) {
        return 0;
    }
    t_flit->computed_routers[this_router_id] = true;

    ComputeVCState& vc_state = vc_compute_state[port_idx][t_flit->vc];

    int opcode = -1;

    if (t_flit->type == 0 || t_flit->type == 10) {
        vc_state.reset();
        vc_state.is_active = true;
        vc_state.compute_op = t_flit->packet->message.compute_op;

        if (!t_flit->packet->message.routing_path.empty() &&
            t_flit->packet->message.routing_path.front() == this_router_id) {

            int heads = (t_flit->packet->message.n_heads > 0) ? t_flit->packet->message.n_heads : 1;
            vc_state.running_max.assign(heads, -1e9);
            vc_state.running_sum.assign(heads, 0.0);
        } else {
            vc_state.running_max = t_flit->packet->message.running_max;
            vc_state.running_sum = t_flit->packet->message.running_sum;
        }
    }

    if (vc_state.is_active) {
        opcode = vc_state.compute_op;
    } else {
        return 0;
    }

    int payload_size = t_flit->get_payload_size();

    if (payload_size > 0 || opcode == ATTENTION) {
        switch(opcode) {
            case MATMUL:
                // MATMUL (15) and LINEAR (0) share the exact same mathematical logic
                // (matrix-vector multiplication for the assigned tasks).
                // Omitting the 'break' here allows MATMUL to cascade directly into
                // the LINEAR block, avoiding unnecessary code duplication.
            case 0:                                         // LINEAR
            {
                int num_tasks = assigned_tasks.size();
                if (num_tasks == 0) break;

                int chunk_offset = t_flit->packet->message.chunk_offset;
                int chunk_row_size = t_flit->packet->message.chunk_row_size;

                for (int t = 0; t < num_tasks; t++) {
                    int task_id = assigned_tasks[t];
                    float local_accum = 0.0f;

                    for (int i = 0; i < payload_size; ++i) {
                        int input_idx = t_flit->global_data_offset + i;

                        if (input_idx >= t_flit->packet->message.psum_offset) continue;

                        if (input_idx >= chunk_offset && input_idx < chunk_offset + chunk_row_size) {
                            int w_offset = (t * chunk_row_size) + (input_idx - chunk_offset);
                            if (w_offset < local_weights.size()) {
                                local_accum += t_flit->get_data(i) * local_weights[w_offset];
                                ++matmul_macs;
                            }
                        }
                    }

                    int target_index = t_flit->packet->message.psum_offset + task_id;
                    if (target_index < t_flit->packet->message.data.size()) {
                        t_flit->packet->message.data[target_index] += local_accum;
                    }
                }
                break;
            }
            case ADD:
            {
                int num_tasks = assigned_tasks.size();
                for (int t = 0; t < num_tasks; t++) {
                    int task_id = assigned_tasks[t];
                    int required_flit_offset = task_id * 2;

                    if (required_flit_offset >= t_flit->global_data_offset &&
                        required_flit_offset + 1 < t_flit->global_data_offset + payload_size) {

                        int local_offset = required_flit_offset - t_flit->global_data_offset;
                        float x = t_flit->get_data(local_offset);
                        float res = t_flit->get_data(local_offset + 1);

                        t_flit->packet->message.data[required_flit_offset] = x + res;
                    }
                }
                break;
            }
            case SWIGLU:
            case GEGLU:
                // Forward gate/up operands; the terminal node applies the nonlinear gate.
                break;
            case ATTENTION:
            {
                // The packet data and partial sums are complete only at the
                // tail.  Earlier flits establish the VC but do not compute.
                if (t_flit->type != 1 && t_flit->type != 10) break;
                if (local_kv_cache.empty()) break;

                int q_dim = t_flit->packet->message.data.size() - t_flit->packet->message.psum_offset;
                int k_dim = (t_flit->packet->message.k_dim > 0) ? t_flit->packet->message.k_dim : q_dim;
                int n_heads = t_flit->packet->message.n_heads;
                if (n_heads <= 0) n_heads = 1;

                int head_dim = q_dim / n_heads;
                int total_kv_heads = std::max(1, k_dim / head_dim);

                int N = t_flit->packet->message.routing_path.size() - 1;
                int router_idx = -1;
                for (int i = 0; i < N; i++) {
                    if (t_flit->packet->message.routing_path[i] == this_router_id) {
                        router_idx = i;
                        break;
                    }
                }

                if (router_idx == -1) break;

                // The running reduction state is updated by the previous
                // router when its tail flit was processed.  The head flit may
                // have arrived earlier, so refresh the state from the packet
                // at the actual computation point.
                if (t_flit->packet->message.running_max.size() == static_cast<size_t>(n_heads)) {
                    vc_state.running_max = t_flit->packet->message.running_max;
                    vc_state.running_sum = t_flit->packet->message.running_sum;
                }

                std::vector<int> valid_physical_slots =
                    getAttentionPhysicalSlots(t_flit, router_idx, N, k_dim);

                int effective_tokens = valid_physical_slots.size();
                if (effective_tokens == 0) break;

                for (int h = 0; h < n_heads; h++) {
                    int kv_h = (total_kv_heads == 1) ? 0 : (h * total_kv_heads / n_heads);

                    std::vector<double> local_scores(effective_tokens, 0.0);
                    double local_max = -1e9;

                    for (int i = 0; i < effective_tokens; i++) {
                        int physical_slot = valid_physical_slots[i];
                        double dot_product = 0.0;
                        int k_offset = physical_slot * (k_dim * 2) + (kv_h * head_dim);

                        for (int d = 0; d < head_dim; d++) {
                            dot_product += (double)t_flit->packet->message.data[(h * head_dim) + d] * (double)local_kv_cache[k_offset + d];
                        }

                        if (attention_head_dim != head_dim) {
                            attention_head_dim = head_dim;
                            attention_score_scale = 1.0 / std::sqrt((double)head_dim);
                        }
                        dot_product *= attention_score_scale;
                        local_scores[i] = dot_product;
                        if (dot_product > local_max) local_max = dot_product;
                    }

                    double m_old = vc_state.running_max[h];
                    double l_old = vc_state.running_sum[h];

                    double m_new = std::max(m_old, local_max);
                    double old_scale = std::exp(m_old - m_new);
                    double local_sum_exp = 0.0;

                    for (int i = 0; i < effective_tokens; i++) {
                        local_scores[i] = std::exp(local_scores[i] - m_new);
                        local_sum_exp += local_scores[i];
                    }
                    double l_new = (l_old * old_scale) + local_sum_exp;

                    for (int d = 0; d < head_dim; d++) {
                        int target_idx = t_flit->packet->message.psum_offset + (h * head_dim) + d;

                        double current_o = (double)t_flit->packet->message.data[target_idx] * old_scale;

                        double local_v_contribution = 0.0;
                        for (int i = 0; i < effective_tokens; i++) {
                            int physical_slot = valid_physical_slots[i];
                            int v_offset = physical_slot * (k_dim * 2) + k_dim + (kv_h * head_dim);
                            local_v_contribution += local_scores[i] * (double)local_kv_cache[v_offset + d];
                        }

                        double final_o = current_o + local_v_contribution;

                        t_flit->packet->message.data[target_idx] = (float)final_o;
                    }

                    vc_state.running_max[h] = m_new;
                    vc_state.running_sum[h] = l_new;
                }

                t_flit->packet->message.running_max = vc_state.running_max;
                t_flit->packet->message.running_sum = vc_state.running_sum;

                break;
            }
        }
    }

    // Clean up the VC state table when the tail flit passes through
    if (t_flit->type == 1 || t_flit->type == 10) {
        vc_state.reset();
    }
    return matmul_macs;
}

RouterAttentionWork VCRouter::finishAttention(Flit* t_flit) {
    if (t_flit->type != 1 && t_flit->type != 10) {
        return {};
    }

    const int this_router_id = id[0] * X_NUM + id[1];
    const int path_length = static_cast<int>(t_flit->packet->message.routing_path.size()) - 1;
    int router_idx = -1;
    for (int r = 0; r < path_length; ++r) {
        if (t_flit->packet->message.routing_path[r] == this_router_id) {
            router_idx = r;
            break;
        }
    }
    if (router_idx < 0) return {};

    const int q_dim = static_cast<int>(t_flit->packet->message.data.size()) -
                      t_flit->packet->message.psum_offset;
    const int k_dim = t_flit->packet->message.k_dim > 0
        ? t_flit->packet->message.k_dim : q_dim;
    const int n_heads = std::max(1, t_flit->packet->message.n_heads);
    const int local_tokens = static_cast<int>(
        getAttentionPhysicalSlots(t_flit, router_idx, path_length, k_dim).size());

    if (router_idx == path_length - 1) {
        const double epsilon = 1e-9;
        const size_t begin = static_cast<size_t>(t_flit->packet->message.psum_offset);
        const size_t end = t_flit->packet->message.data.size();
        for (int h = 0; h < n_heads; ++h) {
            const double denom = h < static_cast<int>(t_flit->packet->message.running_sum.size())
                ? std::max(epsilon, t_flit->packet->message.running_sum[h]) : epsilon;
            const size_t h_begin = begin + static_cast<size_t>(h) * (q_dim / n_heads);
            const size_t h_end = std::min(end, h_begin + static_cast<size_t>(q_dim / n_heads));
            for (size_t p = h_begin; p < h_end; ++p) {
                t_flit->packet->message.data[p] =
                    static_cast<float>(t_flit->packet->message.data[p] / denom);
            }
        }
    }

    return {local_tokens, q_dim, n_heads, router_idx == 0, router_idx == path_length - 1};
}
