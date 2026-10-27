/*
 * Packet.hpp
 *
 */

#ifndef PACKET_HPP_
#define PACKET_HPP_

#include <vector>
#include <stdio.h>
#include "../SimulationTime.hpp"
#include "../PacketTrace.hpp"
#include "MessageKind.hpp"

struct SimulationContext;

struct Message{
  int NI_id;
  int mac_id;
  Cycle out_cycle;
  int slave_id;
  int sequence_id;
  MessageKind kind = MessageKind::DataRequest;
  int data_length;
  int destination;
  int QoS = 0;
  int source_id;
  int signal_id;
  int layer_id = -1;                 // Model layer that generated this packet
  
  std::vector<float> data;          // Contains inputs and partial sums
  int psum_offset = 0;              // Contains the start index of psum in 'data'.
  int k_dim = 0;
  
  int n_heads = 1;
  std::vector<double> running_max;        // Tracks the maximum score (m)
  std::vector<double> running_sum;        // Tracks the sum of exponentials (l)

  int compute_op;                   // type of operation (es. MATMUL, ADD)
  
  std::vector<int> routing_path;    // Source routing: sorted list of routers ID to traverse
  
  //for pooling
  int penable;                       // 0 no, 1 max, 2 avg

  int chunk_offset = 0;
  int chunk_row_size = 0;
  // Output tiling uses batch-local task indices in the compute payload.
  int output_offset = 0;
  int output_count = 0; // Zero retains the legacy full-output/input-slice mode.
};

class Packet
{
public:

  SimulationContext* context = nullptr;
  PacketId uid = 0;
  size_t trace_slot = static_cast<size_t>(-1);
  void mark(PacketEvent event, Cycle time);
  Message message;
  int length;                       // byte length
  ReceiveClass receive_class = ReceiveClass::RequestsAndData;
  int vnet; // Flit/VC indexing representation of VirtualNetwork.
  VirtualNetwork virtualNetwork() const { return static_cast<VirtualNetwork>(vnet); }
  int destination[3];               // x, y, output port of the router

  Cycle send_out_time;              // time of packet sent from PE
  Cycle in_net_time;                // time of packet insert in to the NoC

  void dest_convert(int dest, int router_num_x, int* NI_num);
  int get_next_router_dest();       // source routing helper

  int current_path_index;           // Tracks progress in Source Routing

  // Storage belongs to the simulation; release relinquishes endpoint ownership.
  // Live flits retain the packet until their last reference is released.
  static Packet* allocate(SimulationContext& sim, Message t_message, int router_num_x, int* NI_num);
  static void release(Packet* packet);
private:
  friend class NetworkStorage;
  friend class NI;
  void beginTrace(SimulationContext& sim);
  Packet(const Packet&) = delete;
  Packet& operator=(const Packet&) = delete;
  Packet(Message t_message, int router_num_x, int* NI_num);
  void reset(Message t_message, int router_num_x, int* NI_num);
  size_t live_flits = 0;
  bool release_requested = false;
  bool in_pool = false;
  bool submitted = false;
};

#endif /* PACKET_HPP_ */
