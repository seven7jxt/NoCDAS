/*
 * Packet.cpp
 *
 */

#include "Packet.hpp"
#include "../SimulationContext.hpp"
#include "../parameters.hpp"
#include <iostream>
#include "NetworkStorage.hpp"
//type convert: add change type and vnet

Packet::Packet(Message t_message, int router_num_x, int* NI_num) {
    reset(std::move(t_message), router_num_x, NI_num);
}

// Source Routing helper
int Packet::get_next_router_dest() {
    if (current_path_index < message.routing_path.size()) {
        return message.routing_path[current_path_index];
    }
    return -1;
}

/*
  * @brief convert destination to be x/y/output port of the router
   */
void Packet::dest_convert(int dest, int router_num_x, int* NI_num){
  int hist_num = 0, cur_num = 0, router = 0;
  while (cur_num <= dest){
      hist_num = cur_num;
      cur_num += NI_num[router];
      router++;
  }
  router--;
  destination[0] = router/router_num_x;
  destination[1] = router%router_num_x;
  destination[2] = dest-hist_num;
  //std::cout << destination[0] <<destination[1] << destination[2]<<std::endl;
}

Packet* Packet::allocate(SimulationContext& sim, Message t_message, int router_num_x, int* NI_num) {
    return sim.objects->allocatePacket(std::move(t_message), router_num_x, NI_num);
}

void Packet::release(Packet* packet) {
    if (packet) packet->context->objects->releasePacket(packet);
}

void Packet::reset(Message t_message, int router_num_x, int* NI_num) {
    message = std::move(t_message);
    current_path_index = 0;
    dest_convert(message.destination, router_num_x, NI_num);
    
    send_out_time = 0;
    in_net_time = 0;
    
    MessageKind t_type = message.kind;
    int data_length = message.data_length;
    
    switch (t_type){
        case MessageKind::TerminalOperands: length = data_length * DATA_BYTES + 2; receive_class = ReceiveClass::RequestsAndData; vnet = vnIndex(VirtualNetwork::Memory); break;
        case MessageKind::DataRequest: length = data_length * DATA_BYTES + 2; receive_class = ReceiveClass::RequestsAndData; vnet = vnIndex(VirtualNetwork::Memory); break;
        case MessageKind::DataResponse: length = data_length * DATA_BYTES + 2; receive_class = ReceiveClass::RequestsAndData; vnet = vnIndex(VirtualNetwork::Memory); break;
        case MessageKind::Result: length = data_length * DATA_BYTES + 2; receive_class = ReceiveClass::ResultsAndAcks; vnet = vnIndex(VirtualNetwork::Memory); break;
        case MessageKind::Ack: length = 1 + 2; receive_class = ReceiveClass::ResultsAndAcks; vnet = vnIndex(VirtualNetwork::Memory); break;
        case MessageKind::Distribution: length = data_length * DATA_BYTES + 2; receive_class = ReceiveClass::RequestsAndData; vnet = vnIndex(VirtualNetwork::Compute); break;
        case MessageKind::Compute: length = message.data.size() * DATA_BYTES + 4; receive_class = ReceiveClass::RequestsAndData; vnet = vnIndex(VirtualNetwork::Compute); break;
    }

    if (length % FLIT_LENGTH != 0) {
        length = ((length + FLIT_LENGTH - 1) / FLIT_LENGTH) * FLIT_LENGTH;
    }
}

void Packet::beginTrace(SimulationContext& sim) {
    context = &sim;
    uid = sim.next_packet_id++;
    trace_slot = static_cast<size_t>(-1);
    if (sim.packet_traces.size() < sim.packet_trace_limit) {
        trace_slot = sim.packet_traces.size();
        PacketTrace row;
        row.packet_id = uid;
        row.task_id = message.signal_id;
        row.layer_id = message.layer_id;
        row.message_kind = static_cast<int>(message.kind);
        row.source = message.source_id;
        row.destination = message.destination;
        row.length_bytes = length;
        row.created = sim.cycles;
        row.source_ready = message.out_cycle;
        sim.packet_traces.push_back(row);
    }
}

void Packet::mark(PacketEvent event, Cycle time) {
    if (context && trace_slot < context->packet_traces.size())
        context->packet_traces[trace_slot].mark(event, time);
}
