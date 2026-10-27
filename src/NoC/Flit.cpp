/*
 * flit.cpp
 *
 */

#include "Flit.hpp"
#include "../SimulationContext.hpp"
#include "NetworkStorage.hpp"
#include <stdexcept>
#include "../parameters.hpp"

const int Flit::length = FLIT_LENGTH;
Flit::Flit(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, Cycle t_cycles, int t_pid) {
    trace_node.reserve(X_NUM + Y_NUM);
    trace_time.reserve(X_NUM + Y_NUM);
    reset(t_id, t_type, t_vnet, t_vc, t_packet, t_cycles, t_pid);
}

float Flit::get_data(int local_index) const {
    if (global_data_offset + local_index < packet->message.data.size()) {
        return packet->message.data[global_data_offset + local_index];
    }
    return 0.0f;
}

int Flit::get_payload_size() const {
    return current_payload_size;
}

void Flit::update_data(int local_index, float new_val) {
    if (global_data_offset + local_index < packet->message.data.size()) {
        packet->message.data[global_data_offset + local_index] = new_val;
    }
}

Flit* Flit::allocate(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, Cycle t_cycles, int t_pid) {
    if (!t_packet || !t_packet->context)
        throw std::invalid_argument("Flit requires a simulation-owned packet");
    return t_packet->context->objects->allocateFlit(t_id, t_type, t_vnet, t_vc, t_packet, t_cycles, t_pid);
}

void Flit::release(Flit* flit) {
    if (flit) flit->context->objects->releaseFlit(flit);
}

void Flit::reset(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, Cycle t_cycles, int t_pid) {
    id = t_id;
    type = t_type;
    vnet = t_vnet;
    vc = t_vc;
    out_port = -1;
    packet = t_packet;
    sched_time = t_cycles;
    packetid = t_pid;

    trace_node.clear();
    trace_time.clear();

    computed_routers.reset(); 

    current_payload_size = 0;
    global_data_offset = 0;

    if (packet != nullptr && (packet->message.kind == MessageKind::Distribution || packet->message.kind == MessageKind::Compute)) {
        int elements_per_flit = FLIT_LENGTH / DATA_BYTES;
        global_data_offset = id * elements_per_flit;

        if (global_data_offset >= 0) {
            int remaining_data = packet->message.data.size() - global_data_offset;
            current_payload_size = std::min(elements_per_flit, std::max(0, remaining_data));
        }
    }
}
