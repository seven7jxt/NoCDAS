#include "PacketTrace.hpp"
#include <ostream>
#include <sstream>
#include <locale>

void PacketTrace::mark(PacketEvent event, Cycle time) {
    std::optional<Cycle>* stamp = nullptr;
    switch (event) {
    case PacketEvent::Enqueued: stamp = &enqueued; break;
    case PacketEvent::Packetized: stamp = &packetized; break;
    case PacketEvent::HeadInjected: stamp = &head_injected; break;
    case PacketEvent::TailArrived: stamp = &tail_arrived; break;
    case PacketEvent::Delivered: stamp = &delivered; break;
    case PacketEvent::Consumed: stamp = &consumed; break;
    }
    if (stamp && !*stamp) *stamp = time;
}
void writePacketTraces(std::ostream& destination, const std::vector<PacketTrace>& rows) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "packet_id,task_id,layer_id,message_kind,source,destination,length_bytes,created,source_ready,enqueued,packetized,head_injected,tail_arrived,delivered,consumed\n";
    for (const auto& r : rows) {
        out << r.packet_id << ',' << r.task_id << ',' << r.layer_id << ',' << r.message_kind
            << ',' << r.source << ',' << r.destination << ',' << r.length_bytes
            << ',' << r.created << ',' << r.source_ready;
        for (const auto& stamp : {r.enqueued,r.packetized,r.head_injected,r.tail_arrived,r.delivered,r.consumed}) {
            out << ',';
            if (stamp) out << *stamp;
        }
        out << '\n';
    }
    destination << out.str();
}
