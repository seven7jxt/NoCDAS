/*
 * NI.hpp
 *
 */

#ifndef NI_HPP_
#define NI_HPP_

#include "../SimulationContext.hpp"

#include "FlitBuffer.hpp"
#include "Link.hpp"
#include "PacketBuffer.hpp"
#include "ROutPort.hpp"
#include "VCNetwork.hpp"
#include "NRBase.hpp"
#include "../parameters.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>



class FlitBuffer;
class VCNetwork;
class RInPort;
class PacketBuffer;
class ROutPort;
class Flit;


class NI: public NRBase
{
public:
  SimulationContext& context;
  NI (int t_id, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth, int* t_NI_num);

  int* NI_num;
  int id;
  VCNetwork* vcNetwork;
  int vn_num;
  int vc_per_vn;
  int vc_priority_per_vn;

  Cycle NI_cycle_npooling;

  std::vector<FlitBuffer*> buffer_list;
  std::vector<int> out_vc;
  std::vector<int> state;                               // 0 I; 1 V; 2 A;

  std::vector<int> priority_vc;
  int count_vc;                                         // starvation forbidden
  std::vector<int> priority_switch;
  int count_switch;                                     // starvation forbidden

  int rr_buffer;
  int in_depth;

  int starvation;
  int rr_priority_record;

  ROutPort* out_port;

  RInPort* in_port;



  int num_flit;

  std::uint64_t total_delay;
  int total_num;
  std::uint64_t total_delay_URS;
  int total_num_URS;






  // NI_TX_PACKET_CAPACITY limits waiting packets per VN (0 means unbounded).
  // A successful trySend
  // transfers packet ownership to the NI; submit each allocation once.
  // On a full queue, false leaves ownership and packet timestamps unchanged.
  bool trySend(Packet* packet);
  size_t txQueued(VirtualNetwork network) const;
  size_t txPeak(VirtualNetwork network) const { return tx_peak.at(vnIndex(network)); }
  std::uint64_t txRejectedAttempts() const { return tx_rejected_attempts; }

  // Selective scan preserves legacy scheduling: skipping a not-ready packet
  // does not prevent later eligible packets from being serviced this cycle.
  // Only one cursor may modify a given receive queue at a time.
  class ReceiveCursor {
  public:
    Packet* peek() const;
    void skip();
    Packet* take(); // Transfers ownership to the endpoint; caller releases it.
  private:
    friend class NI;
    ReceiveCursor(NI& owner, ReceiveClass category) : owner_(owner), category_(category) {}
    NI& owner_;
    ReceiveClass category_;
    size_t index_ = 0;
  };
  ReceiveCursor receive(ReceiveClass category) { return ReceiveCursor(*this, category); }

  // send
  bool flitize(Packet*, int);

  void packetDequeue();
  void vcAllocation();
  void switchArbitration();
  void dequeue();

  // receive
  void inputCheck();

  // main
  void runOneStep();

  ~NI();
private:
  std::vector<size_t> tx_peak;
  std::uint64_t tx_rejected_attempts = 0;
  friend class NIPacketizer;
  friend class NIPacketAssembler;
  std::vector<PacketBuffer*> packetBuffer_list;
  std::vector<std::deque<Packet*>> packet_buffer_out;

};

#endif /* NI_HPP_ */
