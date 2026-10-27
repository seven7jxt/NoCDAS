#ifndef NOCDAS_SIMULATION_RESULT_HPP_
#define NOCDAS_SIMULATION_RESULT_HPP_

#include "SimulationTime.hpp"
#include <vector>

enum class StopReason { NotStarted, Completed, CycleLimit };

struct LayerTiming {
    // Normally equal. Legacy fused Conv/Pool execution can span multiple layers.
    int first_layer_id;
    int last_layer_id;
    Cycle start_cycle;
    Cycle end_cycle;
    bool completed;
};

struct SimulationResult {
    StopReason status = StopReason::NotStarted;
    Cycle cycles = 0;
    // Contiguous accounting intervals [start, end), including setup/transition
    // work. A final incomplete interval is included when the cycle limit fires.
    std::vector<LayerTiming> layers;
};

#endif
