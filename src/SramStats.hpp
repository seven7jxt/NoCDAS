#ifndef SRAM_STATS_HPP_
#define SRAM_STATS_HPP_

#include <algorithm>
#include <cstdint>
#include "SimulationTime.hpp"

// Logical live data bytes, not host vector capacity or configured SRAM capacity.
struct SramStats {
    std::uint64_t weight_peak = 0;
    std::uint64_t activation_peak = 0;
    std::uint64_t kv_peak = 0;
    std::uint64_t total_peak = 0;
    std::uint64_t non_kv_peak = 0;
    Cycle non_kv_peak_cycle = 0;
    int non_kv_peak_layer = -1;
    Cycle peak_cycle = 0;
    int peak_layer = -1;

    void observe(std::uint64_t weight, std::uint64_t activation,
                 std::uint64_t kv, Cycle cycle, int layer) {
        weight_peak = std::max(weight_peak, weight);
        activation_peak = std::max(activation_peak, activation);
        kv_peak = std::max(kv_peak, kv);
        const auto non_kv = weight + activation;
        if (non_kv > non_kv_peak) {
            non_kv_peak = non_kv;
            non_kv_peak_cycle = cycle;
            non_kv_peak_layer = layer;
        }
        const auto total = weight + activation + kv;
        if (total > total_peak) {
            total_peak = total;
            peak_cycle = cycle;
            peak_layer = layer;
        }
    }
};

#endif
