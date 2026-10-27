#ifndef NOCDAS_NETWORK_STORAGE_HPP_
#define NOCDAS_NETWORK_STORAGE_HPP_

#include "Packet.hpp"
#include <memory>
#include <vector>
class Flit;

// Host object lifetime only: these pools do not model hardware buffer capacity.
// A packet has one logical owner across TX/network/RX/endpoint. Flits additionally
// retain it. Reuse requires releasePacket() AND the release of every live flit.
class NetworkStorage {
public:
    explicit NetworkStorage(SimulationContext& context);
    ~NetworkStorage();
    NetworkStorage(const NetworkStorage&) = delete;
    NetworkStorage& operator=(const NetworkStorage&) = delete;
    Packet* allocatePacket(Message message, int router_num_x, int* ni_num);
    Flit* allocateFlit(int id, int type, int vnet, int vc, Packet* packet, Cycle time, int task_id);
    void releasePacket(Packet* packet);
    void releaseFlit(Flit* flit);
    size_t livePackets() const { return packets_.size() - free_packets_.size(); }
    size_t liveFlits() const { return flits_.size() - free_flits_.size(); }
    size_t packetSlots() const { return packets_.size(); }
    size_t flitSlots() const { return flits_.size(); }
private:
    void recyclePacket(Packet* packet);
    SimulationContext& context_;
    std::vector<std::unique_ptr<Packet>> packets_;
    std::vector<std::unique_ptr<Flit>> flits_;
    std::vector<Packet*> free_packets_;
    std::vector<Flit*> free_flits_;
};
#endif
