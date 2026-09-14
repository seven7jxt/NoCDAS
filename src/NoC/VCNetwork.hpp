/*
 * VCNetwork.hpp
 *
 */

#ifndef VCNETWORK_HPP_
#define VCNETWORK_HPP_

#include "Link.hpp"
#include "NI.hpp"
#include "VCRouter.hpp"
#include "RInPort.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <map>

extern std::ofstream router_monitor;
extern unsigned int cycles;

class VCRouter;
class NI;

class VCNetwork
{
public:

  /** @brief VCNetwork
   *
   */
  VCNetwork (int router_num, int router_num_x, int NI_num_total, int* NI_num, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth);

  std::vector<VCRouter*> router_list;
  std::vector<NI*> NI_list;


  void runOneStep();

  void clearAllRouterSRAM();
  
  void average_LCS_latency();


  void average_URS_latency();


  int port_num_f(int router);

  // port_utilization in the VC net, for all out ports including NIs
  void port_utilization(int simulate_cycles);

  // port_utilization in the VC net, only consider routers
  void port_utilization_innet(int simulate_cycles);


  void show_LCS_distribution();


  void show_URS_distribution();

  // Traffic counters for router-to-router links.
  // One flit crossing one inter-router link contributes one flit-hop.
  void recordFlitHop(int layer_id);
  std::uint64_t getTotalFlitHops() const;
  std::uint64_t getTotalByteHops() const;
  std::uint64_t getLayerFlitHops(int layer_id) const;
  std::uint64_t getLayerByteHops(int layer_id) const;

//   added
//  void show_VCR_buffer_state();

  int routerNum;
  int NINum;

  ~VCNetwork ();

private:
  std::uint64_t total_flit_hops;
  std::map<int, std::uint64_t> layer_flit_hops;
};

#endif /* VCNETWORK_HPP_ */
