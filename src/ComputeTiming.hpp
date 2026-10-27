#ifndef NOCDAS_COMPUTE_TIMING_HPP_
#define NOCDAS_COMPUTE_TIMING_HPP_

#include "SimulationTime.hpp"
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
Cycle peTransformer(int opcode, int elements, int tile_outputs = 1);
Cycle peAttention(const PEAttentionWork& work);
Cycle peConvOrDense(int operations, bool linear_output);
Cycle peElementwiseBatch(int opcode, size_t index);
Cycle terminalGate();
int routerDistribution(int payload_elements);
int routerOperator(int opcode, int matmul_macs);
int routerAttention(const RouterAttentionWork& work);
}
#endif
