#ifndef NOCDAS_PACKET_TRACE_HPP_
#define NOCDAS_PACKET_TRACE_HPP_
#include "SimulationTime.hpp"
#include <optional>
#include <vector>
#include <iosfwd>

using PacketId = std::uint64_t;
enum class PacketEvent { Enqueued, Packetized, HeadInjected, TailArrived, Delivered, Consumed };

// Unique ID per physical packet allocation; task_id may repeat across refills.
// Missing events remain null, including packets still in flight at termination.
struct PacketTrace {
    PacketId packet_id = 0;
    int task_id = -1, layer_id = -1, message_kind = -1;
    int source = -1, destination = -1, length_bytes = 0;
    Cycle created = 0, source_ready = 0;
    std::optional<Cycle> enqueued, packetized, head_injected;
    std::optional<Cycle> tail_arrived, delivered, consumed;
    void mark(PacketEvent event, Cycle time);
};
void writePacketTraces(std::ostream& out, const std::vector<PacketTrace>& rows);
#endif
