#ifndef NOCDAS_SIMULATION_TIME_HPP_
#define NOCDAS_SIMULATION_TIME_HPP_

#include <cstdint>

// Absolute cycle timestamps must remain exact throughout long simulations.
using Cycle = std::uint64_t;

// Convert a duration, without rounding or aligning its start to a clock edge.
constexpr Cycle toNoCCycles(Cycle local_cycles, Cycle noc_cycles_per_local_cycle) {
    return local_cycles * noc_cycles_per_local_cycle;
}

#endif
