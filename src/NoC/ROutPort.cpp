/*
 * ROutPort.cpp
 *
 *  Created on: 2019年8月22日
 *      Author: wr
 */

#include "ROutPort.hpp"

ROutPort::ROutPort (SimulationContext& sim, int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth)
   : Port(sim, t_id, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_depth)
{
  out_link = NULL;
}

ROutPort::~ROutPort(){

}

