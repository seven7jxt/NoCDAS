#ifndef NOCDAS_COMPUTE_RESOURCES_HPP_
#define NOCDAS_COMPUTE_RESOURCES_HPP_

#include "ComputeUnit.hpp"
#include "parameters.hpp"
#include <functional>
#include <memory>

struct PEComputeConfig {
    int ops_per_cycle = PE_NUM_OP;
    Cycle clock_ratio = PE_FREQ_RATIO; // NoC cycles per PE cycle.
    int gate_latency = PE_GATE_LATENCY; // Per scalar in a serial gate batch.
};

struct RouterComputeConfig {
    double macs_per_cycle = ROUTER_MACS_PER_CYCLE; // Router runs on the NoC clock.
    int sram_width_bits = ROUTER_SRAM_WIDTH;
};

struct TerminalComputeConfig {
    int lanes = 0; // Zero derives the total from the number of non-MC PEs.
    Cycle clock_ratio = PE_FREQ_RATIO;
    int gate_latency = PE_GATE_LATENCY;
};

enum class ComputeRole { PE, Router, Terminal };
struct ComputeLocation {
    ComputeRole role;
    int node_id; // NI/router ID; for a terminal this is its MC's NI ID.
    int index = 0; // Local PE or terminal lane index; zero for a router.
};

// Called once per resource during topology construction, never on the cycle path.
using ComputeUnitFactory = std::function<std::unique_ptr<ComputeUnit>(ComputeLocation)>;

struct ComputeResources {
    PEComputeConfig pe;
    RouterComputeConfig router;
    TerminalComputeConfig terminal;
    ComputeUnitFactory factory;

    std::unique_ptr<ComputeUnit> makeUnit(ComputeLocation location) const {
        return factory ? factory(location) : std::make_unique<SerialComputeUnit>();
    }
};

#endif
