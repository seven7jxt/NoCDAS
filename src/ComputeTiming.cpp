#include "ComputeTiming.hpp"
#include "Model.hpp"
#include <algorithm>
#include <cmath>

namespace LegacyComputeTiming {
Cycle peTransformer(int opcode, int elements, int tile_outputs) {
    int cycles = 0;
    switch (opcode) {
    case MATMUL: {
        const int systolic_dim = std::ceil(std::sqrt(PE_NUM_OP));
        cycles = (systolic_dim + elements * tile_outputs / PE_NUM_OP + 1) * MAC_LATENCY;
        break;
    }
    case LAYERNORM:
        cycles = (5 * elements / PE_NUM_OP + 1) * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
        break;
    case SOFTMAX_TR:
        cycles = (3 * elements / PE_NUM_OP + 1) * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
        break;
    case RMSNORM:
        cycles = (3 * elements / PE_NUM_OP + 1) * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
        break;
    case ADD: cycles = ADD_LATENCY; break;
    case EMBEDDING: cycles = 1; break;
    case SWIGLU:
    case GEGLU:
        cycles = ((4 + PE_NUM_OP - 1) / PE_NUM_OP) * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
        break;
    case ROPE: cycles = 2 * MAC_LATENCY + CORDIC_LATENCY; break;
    }
    return toNoCCycles(cycles, PE_FREQ_RATIO);
}

Cycle peAttention(const PEAttentionWork& work) {
    const int value_ops = work.history / PE_NUM_OP + 1;
    int cycles = value_ops * MAC_LATENCY;
    if (work.rebuild_scores) {
        const int dot_ops = work.history * work.head_dim / PE_NUM_OP + 1;
        const int softmax_ops = 3 * work.history / PE_NUM_OP + 1;
        cycles = (dot_ops + softmax_ops + value_ops) * MAC_LATENCY +
                 work.history * EXP_LATENCY + DIV_LATENCY + CORDIC_LATENCY;
    }
    return toNoCCycles(cycles, PE_FREQ_RATIO);
}

Cycle peConvOrDense(int operations, bool linear_output) {
    // Legacy linear output removes one PE cycle from the final chunk only.
    return toNoCCycles(operations / PE_NUM_OP + (linear_output ? 0 : 1), PE_FREQ_RATIO);
}

Cycle peElementwiseBatch(int opcode, size_t index) {
    const Cycle cycles = opcode == ADD
        ? ((static_cast<Cycle>(index) + PE_NUM_OP) / PE_NUM_OP) * ADD_LATENCY
        : static_cast<Cycle>(index + 1) * PE_GATE_LATENCY;
    return toNoCCycles(cycles, PE_FREQ_RATIO);
}

Cycle terminalGate() {
    return toNoCCycles(PE_GATE_LATENCY, PE_FREQ_RATIO);
}

int routerDistribution(int payload_elements) {
    const int elements_per_cycle = ROUTER_SRAM_WIDTH / (DATA_BYTES * 8);
    return std::max(1, (payload_elements + elements_per_cycle - 1) / elements_per_cycle);
}

int routerOperator(int opcode, int macs) {
    if (opcode == SWIGLU || opcode == GEGLU) return 1;
    if (opcode == ADD) return ADD_LATENCY;
    if (opcode != MATMUL && opcode != LINEAR) return MAC_LATENCY;
    const int mac_cycles = static_cast<int>(std::ceil(macs / static_cast<double>(ROUTER_MACS_PER_CYCLE)));
    const int elements_per_cycle = ROUTER_SRAM_WIDTH / (DATA_BYTES * 8);
    const int sram_cycles = (macs + elements_per_cycle - 1) / elements_per_cycle;
    // SRAM weight reads overlap MAC issue; a no-work flit still pays forwarding.
    return macs > 0 ? std::max(mac_cycles, sram_cycles) + MAC_LATENCY - 1 : 1;
}

int routerAttention(const RouterAttentionWork& work) {
    int delay = 1;
    if (work.local_tokens > 0) {
        const int macs = work.local_tokens * work.q_dim * 2 + 3 * work.local_tokens * work.n_heads;
        const int mac_cycles = static_cast<int>(std::ceil(macs / static_cast<double>(ROUTER_MACS_PER_CYCLE)));
        const int exp_ops = work.n_heads * (work.local_tokens + 1);
        // One EXP pipeline, II=1; the router-wide MFU stays reserved for the batch.
        delay = mac_cycles * MAC_LATENCY + EXP_LATENCY + exp_ops - 1;
    }
    if (work.last)
        delay += DIV_LATENCY + static_cast<int>(std::ceil(work.q_dim / static_cast<double>(ROUTER_MACS_PER_CYCLE))) * MAC_LATENCY;
    if (work.first) delay += SQRT_LATENCY;
    return delay;
}
}
