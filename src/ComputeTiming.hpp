#ifndef NOCDAS_COMPUTE_TIMING_HPP_
#define NOCDAS_COMPUTE_TIMING_HPP_

#include "SimulationTime.hpp"
#include "ComputeResources.hpp"
#include <cstddef>

// Work observed by numerical attention execution; no scheduler state.
struct PEAttentionWork {
    int history;
    int head_dim;
    bool rebuild_scores;
};
struct RouterAttentionWork {
    int local_tokens = 0;
    int q_dim = 0;
    int n_heads = 1;
    bool first = false;
    bool last = false;
};

// Existing formulas, including their original rounding and pipeline assumptions.
// Every returned duration is in NoC cycles. No function advances simulation time.
namespace LegacyComputeTiming {
Cycle peTransformer(int opcode, int elements, int tile_outputs = 1, const PEComputeConfig& config = {});
Cycle peAttention(const PEAttentionWork& work, const PEComputeConfig& config = {});
Cycle peConvOrDense(int operations, bool linear_output, const PEComputeConfig& config = {});
Cycle peElementwiseBatch(int opcode, size_t index, const PEComputeConfig& config = {});
Cycle terminalGate(const TerminalComputeConfig& config = {});
int routerDistribution(int payload_elements, const RouterComputeConfig& config = {});
int routerOperator(int opcode, int matmul_macs, const RouterComputeConfig& config = {});
int routerAttention(const RouterAttentionWork& work, const RouterComputeConfig& config = {});
}
#endif
