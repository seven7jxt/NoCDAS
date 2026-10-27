#ifndef NOCDAS_COMPUTE_KERNELS_HPP_
#define NOCDAS_COMPUTE_KERNELS_HPP_

#include "Model.hpp"
#include <cmath>

namespace ComputeKernels {
// Keep serial accumulation and float intermediates identical to the legacy PE.
inline float dot(const float* input, const float* weights, int count, float sum = 0.0f) {
    for (int i = 0; i < count; ++i) sum += input[i] * weights[i];
    return sum;
}
inline float pool(const float* input, int count, bool average) {
    float result = input[0];
    for (int i = 1; i < count; ++i) {
        if (average) result = result + input[i];
        else if (input[i] > result) result = input[i];
    }
    return average ? result / count : result;
}
// PE batches and MC terminal lanes use the same expressions. Scalar PE gates
// retain their separate intermediate rounding in MAC::computeTransformerElement.
inline float elementwisePair(int opcode, float gate, float up) {
    return opcode == ADD ? gate + up : opcode == SWIGLU
        ? gate * (1.0 / (1.0 + std::exp(-gate))) * up
        : 0.5f * gate * (1.0f + std::erf(gate / 1.41421356f)) * up;
}
}
#endif
