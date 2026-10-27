#include "Simulation.hpp"
#include "MACnet.hpp"
#include "Model.hpp"
#include "NoC/VCNetwork.hpp"
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

#ifndef NOCDAS_BUILD_TYPE
#define NOCDAS_BUILD_TYPE "unspecified"
#endif

namespace {
// std::quoted is not JSON escaping (e.g. embedded newlines in input filenames).
std::string jsonString(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0') << static_cast<unsigned>(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
const char* stopName(StopReason reason) {
    switch (reason) {
    case StopReason::Completed: return "completed";
    case StopReason::CycleLimit: return "cycle_limit";
    default: return "not_started";
    }
}
const char* layerName(char kind) {
    switch (kind) {
    case 'i': return "Input";
    case 'c': return "Conv2D";
    case 'f': return "Dense";
    case 'p': return "Pool";
    case 'e': return "Embedding";
    case 'm': return "MatMul";
    case 'l': return "LayerNorm";
    case 'r': return "RMSNorm";
    case 's': return "Softmax";
    case 'a': return "Add";
    case 'w': return "SwiGLU";
    case 'g': return "GeGLU";
    case 'o': return "RoPE";
    case 't': return "Attention";
    default: return "Unknown";
    }
}
}

void Simulation::writeJson(std::ostream& destination) const {
    if (!macnet) throw std::logic_error("Simulation has not started");
    // Keep report formatting independent of the caller's stream flags/locale.
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
#ifdef cNoC_MODE
    const char* mode = "cnoc";
#else
    const char* mode = "baseline";
#endif
    const auto waits = vcNetwork->getTotalWaitCycles();
    std::uint64_t flits = 0;
    for (const auto* ni : vcNetwork->NI_list) flits += ni->num_flit;
    const auto& ns = context_.ni_statistics;
    out << "{\n  \"schema_version\": 1,\n  \"mode\": " << jsonString(mode)
        << ",\n  \"status\": " << jsonString(stopName(result_.status))
        << ",\n  \"completed\": " << (result_.status == StopReason::Completed ? "true" : "false")
        << ",\n  \"cycle_unit\": \"noc_clock\",\n  \"cycles\": " << result_.cycles
        << ",\n  \"max_cycles\": " << options_.max_cycles
        << ",\n  \"build\": {\"type\": " << jsonString(NOCDAS_BUILD_TYPE)
        << ", \"compiler\": " << jsonString(__VERSION__) << "},"
        << "\n  \"workload\": {\"model\": " << jsonString(options_.model_file)
        << ", \"weights\": " << jsonString(options_.weight_file)
        << ", \"input\": " << jsonString(options_.input_file) << "},"
        << "\n  \"config\": {\"mesh_x\": " << X_NUM << ", \"mesh_y\": " << Y_NUM
        << ", \"nodes\": " << TOT_NUM << ", \"mc_nodes\": " << MEM_NODES
        << ", \"noc_frequency_ghz\": " << FREQUENCY
        << ", \"pe_frequency_ratio\": " << PE_FREQ_RATIO
        << ", \"pe_ops_per_cycle\": " << PE_NUM_OP
        << ", \"router_macs_per_cycle\": " << ROUTER_MACS_PER_CYCLE
        << ", \"router_sram_width_bits\": " << ROUTER_SRAM_WIDTH
        << ", \"flit_bytes\": " << FLIT_LENGTH << ", \"data_bytes\": " << DATA_BYTES
        << ", \"vn_count\": " << VN_NUM << ", \"vc_per_vn\": " << VC_PER_VN
        << ", \"priority_vc_per_vn\": " << VC_PRIORITY_PER_VN;
    const int vc_depth = INPORT_FLIT_BUFFER_SIZE;
    out << ", \"router_vc_depth_flits\": " << vc_depth
        << ", \"ni_tx_fifo_depth_setting\": " << NI_TX_FIFO_DEPTH
        << ", \"ni_tx_packet_capacity_per_vn\": " << NI_TX_PACKET_CAPACITY
        << ", \"ni_rx_fifo_depth_setting\": " << NI_RX_FIFO_DEPTH
        << ", \"ni_tx_flit_depth\": " << NI_TX_VC_DEPTH_FLITS
        << ", \"ni_rx_flit_depth\": " << NI_RX_VC_DEPTH_FLITS
        << ", \"pe_weight_sram_bytes\": " << MAC_WEIGHT_SRAM_LIMIT * DATA_BYTES
        << ", \"pe_input_sram_bytes\": " << MAC_INPUT_SRAM_LIMIT * DATA_BYTES
        << ", \"pe_kv_cache_bytes\": " << KV_CACHE_SIZE * DATA_BYTES
        << ", \"kv_cache_enabled\": " << ENABLE_KV_CACHE
        << ", \"router_sram_bytes_setting\": " << ROUTER_SRAM_LIMIT * DATA_BYTES
        << ", \"link_cycles\": " << LINK_TIME
        << ", \"memory_read_cycles_per_element\": " << MEM_read_delay
        << ", \"cache_delay_cycles\": " << CACHE_DELAY
        << ", \"mc_node_ids\": [";
    for (int i = 0; i < MEM_NODES; ++i) { if (i) out << ','; out << dest_list[i]; }
    out << "]},\n  \"traffic\": {\"legacy_packet_id_base\": " << context_.packet_id_base
        << ", \"packets_packetized\": " << static_cast<std::uint64_t>(ns.count_s) + ns.count_s_resp
        << ", \"packets_delivered_to_ni_queue\": " << static_cast<std::uint64_t>(ns.count_r) + ns.count_r_resp
        << ", \"flits_packetized\": " << flits
        << ", \"inter_router_flit_hops\": " << vcNetwork->getTotalFlitHops()
        << ", \"inter_router_byte_hops\": " << vcNetwork->getTotalByteHops() << "},"
        << "\n  \"legacy_wait_counters\": {\"pe_network\": " << waits.pe_network
        << ", \"pe_memory\": " << waits.pe_memory << ", \"router_mfu\": " << waits.router_mfu
        << ", \"router_kv\": " << waits.router_kv << "},"
        << "\n  \"kv_evictions\": " << vcNetwork->getTotalKVEvictions()
        << ",\n  \"layer_interval_semantics\": \"[start,end), includes setup and transitions; incomplete interval included on cycle_limit\","
        << "\n  \"layers\": [";
    for (size_t i = 0; i < result_.layers.size(); ++i) {
        const auto& row = result_.layers[i];
        if (i) out << ',';
        out << "\n    {\"first_layer_id\": " << row.first_layer_id
            << ", \"last_layer_id\": " << row.last_layer_id
            << ", \"type\": " << jsonString(layerName(cnnmodel->all_layer_type[row.first_layer_id]))
            << ", \"start_cycle\": " << row.start_cycle << ", \"end_cycle\": " << row.end_cycle
            << ", \"duration_cycles\": " << row.end_cycle - row.start_cycle
            << ", \"completed\": " << (row.completed ? "true" : "false") << '}';
    }
    out << "\n  ],\n  \"accounting\": {"
        << "\"wait_units\": \"PE-cycles for PE waits; output-port-cycles for legacy router waits\","
        << "\"latency_additive\": false,"
        << "\"router_service_scope\": \"MFU/distribution reservation, excluding KV retry hold\","
        << "\"pe_network_wait_scope\": \"request-to-response wait minus assigned MC service; includes queueing and endpoint delays\","
        << "\"packet_trace_limit\": " << context_.packet_trace_limit
        << ",\"packets_created\": " << context_.next_packet_id
        << ",\"packets_traced\": " << context_.packet_traces.size()
        << ",\"pe\": [";
    for (size_t i=0; i<vcNetwork->peAccounting().size(); ++i) {
        if (i) out << ',';
        const auto& p = vcNetwork->peAccounting()[i];
        out << "{\"id\":" << i << ",\"network_wait_cycles\":" << p.network_wait_cycles
            << ",\"memory_service_wait_cycles\":" << p.memory_service_wait_cycles << '}';
    }
    out << "],\"router\": [";
    for (size_t i=0; i<vcNetwork->routerAccounting().size(); ++i) {
        if (i) out << ',';
        const auto& r = vcNetwork->routerAccounting()[i];
        out << "{\"id\":" << i << ",\"blocked_port_cycles\":" << r.blocked_port_cycles
            << ",\"blocked_router_cycles\":" << r.blocked_router_cycles
            << ",\"service_occupied_cycles\":" << r.service_occupied_cycles
            << ",\"kv_hold_cycles\":" << r.kv_hold_cycles << '}';
    }
    out << "]},\n  \"ni_tx\": [";
    for (size_t i = 0; i < vcNetwork->NI_list.size(); ++i) {
        if (i) out << ',';
        const auto* ni = vcNetwork->NI_list[i];
        out << "{\"id\":" << ni->id
            << ",\"rejected_send_attempts\":" << ni->txRejectedAttempts()
            << ",\"peak_queued_packets_per_vn\":[";
        for (int vn = 0; vn < ni->vn_num; ++vn) {
            if (vn) out << ',';
            out << ni->txPeak(static_cast<VirtualNetwork>(vn));
        }
        out << "]}";
    }
    out << "]\n}\n";
    destination << out.str();
}

void Simulation::writeLayerCsv(std::ostream& destination) const {
    if (!macnet) throw std::logic_error("Simulation has not started");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "first_layer_id,last_layer_id,type,start_cycle,end_cycle,duration_cycles,completed\n";
    for (const auto& row : result_.layers) {
        out << row.first_layer_id << ',' << row.last_layer_id << ','
            << layerName(cnnmodel->all_layer_type[row.first_layer_id]) << ','
            << row.start_cycle << ',' << row.end_cycle << ','
            << row.end_cycle - row.start_cycle << ',' << (row.completed ? "true" : "false") << '\n';
    }
    destination << out.str();
}
