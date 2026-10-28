#include "ComputeTiming.hpp"
#include "Model.hpp"
#include <algorithm>
#include <cmath>

namespace LegacyComputeTiming {
Cycle peTransformer(int opcode, int elements, int tile_outputs, const PEComputeConfig& config) {
    int cycles = 0;
    switch (opcode) {
    case MATMUL: {
        const int systolic_dim = std::ceil(std::sqrt(config.ops_per_cycle));
        cycles = (systolic_dim + elements * tile_outputs / config.ops_per_cycle + 1) * MAC_LATENCY;
        break;
    }
    case LAYERNORM:
        cycles = (5 * elements / config.ops_per_cycle + 1) * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
        break;
    case SOFTMAX_TR:
        cycles = (3 * elements / config.ops_per_cycle + 1) * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
        break;
    case RMSNORM:
        cycles = (3 * elements / config.ops_per_cycle + 1) * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
        break;
    case ADD: cycles = ADD_LATENCY; break;
    case EMBEDDING: cycles = 1; break;
    case SWIGLU:
    case GEGLU:
        cycles = ((4 + config.ops_per_cycle - 1) / config.ops_per_cycle) * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
        break;
    case ROPE: cycles = 2 * MAC_LATENCY + CORDIC_LATENCY; break;
    }
    return toNoCCycles(cycles, config.clock_ratio);
}

Cycle peAttention(const PEAttentionWork& work, const PEComputeConfig& config) {
    const int value_ops = work.history / config.ops_per_cycle + 1;
    int cycles = value_ops * MAC_LATENCY;
    if (work.rebuild_scores) {
        const int dot_ops = work.history * work.head_dim / config.ops_per_cycle + 1;
        const int softmax_ops = 3 * work.history / config.ops_per_cycle + 1;
        cycles = (dot_ops + softmax_ops + value_ops) * MAC_LATENCY +
                 work.history * EXP_LATENCY + DIV_LATENCY + CORDIC_LATENCY;
    }
    return toNoCCycles(cycles, config.clock_ratio);
}

Cycle peConvOrDense(int operations, bool linear_output, const PEComputeConfig& config) {
    // Legacy linear output removes one PE cycle from the final chunk only.
    return toNoCCycles(operations / config.ops_per_cycle + (linear_output ? 0 : 1), config.clock_ratio);
}

Cycle peElementwiseBatch(int opcode, size_t index, const PEComputeConfig& config) {
    const Cycle cycles = opcode == ADD
        ? ((static_cast<Cycle>(index) + config.ops_per_cycle) / config.ops_per_cycle) * ADD_LATENCY
        : static_cast<Cycle>(index + 1) * config.gate_latency;
    return toNoCCycles(cycles, config.clock_ratio);
}

Cycle terminalGate(const TerminalComputeConfig& config) {
    return toNoCCycles(config.gate_latency, config.clock_ratio);
}

int routerDistribution(int payload_elements, const RouterComputeConfig& config) {
    const int elements_per_cycle = config.sram_width_bits / (DATA_BYTES * 8);
    return std::max(1, (payload_elements + elements_per_cycle - 1) / elements_per_cycle);
}

int routerOperator(int opcode, int macs, const RouterComputeConfig& config) {
    if (opcode == SWIGLU || opcode == GEGLU) return 1;
    if (opcode == ADD) return ADD_LATENCY;
    if (opcode != MATMUL && opcode != LINEAR) return MAC_LATENCY;
    const int mac_cycles = static_cast<int>(std::ceil(macs / static_cast<double>(config.macs_per_cycle)));
    const int elements_per_cycle = config.sram_width_bits / (DATA_BYTES * 8);
    const int sram_cycles = (macs + elements_per_cycle - 1) / elements_per_cycle;
    // SRAM weight reads overlap MAC issue; a no-work flit still pays forwarding.
    return macs > 0 ? std::max(mac_cycles, sram_cycles) + MAC_LATENCY - 1 : 1;
}

int routerAttention(const RouterAttentionWork& work, const RouterComputeConfig& config) {
    int delay = 1;
    if (work.local_tokens > 0) {
        const int macs = work.local_tokens * work.q_dim * 2 + 3 * work.local_tokens * work.n_heads;
        const int mac_cycles = static_cast<int>(std::ceil(macs / static_cast<double>(config.macs_per_cycle)));
        const int exp_ops = work.n_heads * (work.local_tokens + 1);
        // One EXP pipeline, II=1; the router-wide MFU stays reserved for the batch.
        delay = mac_cycles * MAC_LATENCY + EXP_LATENCY + exp_ops - 1;
    }
    if (work.last)
        delay += DIV_LATENCY + static_cast<int>(std::ceil(work.q_dim / static_cast<double>(config.macs_per_cycle))) * MAC_LATENCY;
    if (work.first) delay += SQRT_LATENCY;
    return delay;
}
}
