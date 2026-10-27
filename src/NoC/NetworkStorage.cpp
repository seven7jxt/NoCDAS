#include "NetworkStorage.hpp"
#include "Flit.hpp"
#include "../SimulationContext.hpp"
#include <stdexcept>

NetworkStorage::NetworkStorage(SimulationContext& context) : context_(context) {}
NetworkStorage::~NetworkStorage() = default;

Packet* NetworkStorage::allocatePacket(Message message, int router_num_x, int* ni_num) {
    Packet* packet;
    if (free_packets_.empty()) {
        auto owned = std::unique_ptr<Packet>(new Packet(std::move(message), router_num_x, ni_num));
        packet = owned.get();
        packets_.push_back(std::move(owned));
    } else {
        packet = free_packets_.back();
        free_packets_.pop_back();
        packet->reset(std::move(message), router_num_x, ni_num);
    }
    packet->live_flits = 0;
    packet->release_requested = false;
    packet->in_pool = false;
    packet->submitted = false;
    packet->beginTrace(context_);
    return packet;
}

Flit* NetworkStorage::allocateFlit(int id, int type, int vnet, int vc,
                                 Packet* packet, Cycle time, int task_id) {
    if (!packet || packet->context != &context_ || packet->release_requested || packet->in_pool)
        throw std::logic_error("Cannot retain an inactive or foreign packet");
    Flit* flit;
    if (free_flits_.empty()) {
        auto owned = std::unique_ptr<Flit>(new Flit(id, type, vnet, vc, packet, time, task_id));
        flit = owned.get();
        flits_.push_back(std::move(owned));
    } else {
        flit = free_flits_.back();
        free_flits_.pop_back();
        flit->reset(id, type, vnet, vc, packet, time, task_id);
    }
    flit->context = &context_;
    flit->in_pool = false;
    ++packet->live_flits;
    return flit;
}

void NetworkStorage::recyclePacket(Packet* packet) {
    if (packet->release_requested && packet->live_flits == 0) {
        packet->in_pool = true;
        free_packets_.push_back(packet);
    }
}

void NetworkStorage::releasePacket(Packet* packet) {
    if (!packet || packet->context != &context_ || packet->release_requested || packet->in_pool)
        throw std::logic_error("Duplicate or foreign packet release");
    packet->release_requested = true;
    recyclePacket(packet);
}

void NetworkStorage::releaseFlit(Flit* flit) {
    if (!flit || flit->context != &context_ || flit->in_pool)
        throw std::logic_error("Duplicate or foreign flit release");
    Packet* packet = flit->packet;
    if (!packet || packet->live_flits == 0 || packet->in_pool)
        throw std::logic_error("Flit refers to an inactive packet");
    --packet->live_flits;
    flit->packet = nullptr;
    flit->in_pool = true;
    free_flits_.push_back(flit);
    recyclePacket(packet);
}
