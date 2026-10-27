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

  std::vector<PacketBuffer*> packetBuffer_list;
  std::vector<std::deque<Packet*> > packet_buffer_out;  // 0 request; 1 response
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

};

#endif /* NI_HPP_ */
