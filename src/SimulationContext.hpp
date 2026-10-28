#ifndef NOCDAS_SIMULATION_CONTEXT_HPP_
#define NOCDAS_SIMULATION_CONTEXT_HPP_

#include "SimulationTime.hpp"
#include "ComputeResources.hpp"
#include "parameters.hpp"
#include <vector>
#include <memory>

class NetworkStorage;
#include "PacketTrace.hpp"

struct NIStatistics {
    int count_s = {};
    int count_r = {};
    int count_s_resp = {};
    int count_r_resp = {};
    int count_input = {};
    std::int64_t worst_LCS = {};
    std::int64_t worst_URS = {};
    int URS_delay_distribution[DISTRIBUTION_NUM] = {};
    int LCS_delay_distribution[DISTRIBUTION_NUM] = {};
    int URS_latency_single_dis[100] = {};
    int URS_latency_single_count = {};
    int URS_latency_single_total = {};
    int URS_latency_single_worst = {};
    int LCS_latency_single_dis[100] = {};
    int LCS_latency_single_count = {};
    int LCS_latency_single_total = {};
    int LCS_latency_single_worst = {};
    int converse_latency_single_dis[100] = {};
    int converse_latency_single_count = {};
    int converse_latency_single_total = {};
    int converse_latency_single_worst = {};
};

// One clock per simulation. Timestamps retain the legacy NoC-cycle units.
struct SimulationContext {
    SimulationContext();
    ~SimulationContext();
    SimulationContext(const SimulationContext&) = delete;
    SimulationContext& operator=(const SimulationContext&) = delete;
    // Outlives the network and releases all backing storage, including aborted traffic.
    std::unique_ptr<NetworkStorage> objects;
    Cycle cycles = 0;
    ComputeResources compute;
    // Legacy task-based packet ID base; retain its allocation scheme in this phase.
    int packet_id_base = 0;
    PacketId next_packet_id = 0;
    size_t packet_trace_limit = CountNum;
    std::vector<PacketTrace> packet_traces;
    NIStatistics ni_statistics;
};

#endif
