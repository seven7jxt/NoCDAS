#include "NIPacketizer.hpp"
#include "NI.hpp"

bool NIPacketizer::packetize(NI& ni, Packet* packet, int vn) {
  int base;
  //added to replace global packet_id
  int pid = packet->message.signal_id;
  // for priority packet with individual VCs
  if(packet->message.QoS == 3){
      base = ni.vn_num*ni.vc_per_vn+vn*ni.vc_priority_per_vn;
      for(int i=base; i<base+ni.vc_priority_per_vn; i++){
	  if(ni.state[i] == 0){  // idle
	      //packet_id++;  // if success
	      int length = (packet->length-1)/FLIT_LENGTH+1; //round up
	      packet->mark(PacketEvent::Packetized, ni.context.cycles);
          packet->send_out_time = ni.context.cycles;
	      ni.num_flit += length;
	      if(length == 1){
		  Flit* flit = Flit::allocate(0, 10, vn, i, packet, ni.context.cycles, pid);
		  flit->trace_node.push_back(ni.id);
		  flit->trace_time.push_back(ni.context.cycles);
		  flit->packet->in_net_time = ni.context.cycles;
		  ni.buffer_list[i]->enqueue(flit);
	      }else{
		  for(int id=0; id<length; id++){
		      if(id == 0){
			  Flit* flit = Flit::allocate(id, 0, vn, i, packet, ni.context.cycles, pid);
			  flit->trace_node.push_back(ni.id);
			  flit->trace_time.push_back(ni.context.cycles);
			  flit->packet->in_net_time = ni.context.cycles;
			  ni.buffer_list[i]->enqueue(flit);
		      }else
			if(id == length-1){
			    Flit* flit = Flit::allocate(id, 1, vn, i, packet, ni.context.cycles, pid);
			    flit->trace_node.push_back(ni.id);
			    flit->trace_time.push_back(ni.context.cycles);
			    flit->packet->in_net_time = ni.context.cycles;
			    ni.buffer_list[i]->enqueue(flit);
			}else{
			    Flit* flit = Flit::allocate(id, 2, vn, i, packet, ni.context.cycles, pid);
			    flit->trace_node.push_back(ni.id);
			    flit->trace_time.push_back(ni.context.cycles);
			    flit->packet->in_net_time = ni.context.cycles;
			    ni.buffer_list[i]->enqueue(flit);
			}
		  }
	      }
	      ni.state[i] = 1;  // wait for VC allocation
	      return true;
	  }
      }
      return false;
  }

  // for normal packet or priority packet with shared VCs
  else{
#ifdef SHARED_PRI
      if(packet->message.QoS == 1){ // for LCS to use all the VCs
#endif

      base = vn*ni.vc_per_vn; // 0 or 2  -> buffer_list(0,1) or buffer_list(2,3)
      for(int i=base; i<base+ni.vc_per_vn; i++){
	  if(ni.state[i] == 0){  // idle
	      //packet_id++;  // if success
	      packet->mark(PacketEvent::Packetized, ni.context.cycles);
          packet->send_out_time = ni.context.cycles;
	      int length = (packet->length-1)/FLIT_LENGTH + 1; //round up
	      ni.num_flit += length;
	      if(length == 1){
		  Flit* flit = Flit::allocate(0, 10, vn, i, packet, ni.context.cycles, pid);
		  flit->trace_node.push_back(ni.id);
		  flit->trace_time.push_back(ni.context.cycles);
		  flit->packet->in_net_time = ni.context.cycles;
		  ni.buffer_list[i]->enqueue(flit);
		  if(flit->packet->message.QoS == 1){
		      ni.priority_vc.push_back(i);
		      ni.priority_switch.push_back(i);
		  }
	      }else{
		  for(int id=0; id<length; id++){
		      if(id == 0){
			  Flit* flit = Flit::allocate(id, 0, vn, i, packet, ni.context.cycles, pid);
			  flit->trace_node.push_back(ni.id);
			  flit->trace_time.push_back(ni.context.cycles);
			  flit->packet->in_net_time = ni.context.cycles;
			  ni.buffer_list[i]->enqueue(flit);
			  if(flit->packet->message.QoS == 1){
			      ni.priority_vc.push_back(i);
			      ni.priority_switch.push_back(i);
			  }
		      }else
			if(id == length-1){
			    Flit* flit = Flit::allocate(id, 1, vn, i, packet, ni.context.cycles, pid);
			    flit->trace_node.push_back(ni.id);
			    flit->trace_time.push_back(ni.context.cycles);
			    flit->packet->in_net_time = ni.context.cycles;
			    ni.buffer_list[i]->enqueue(flit);
			}else{
			    Flit* flit = Flit::allocate(id, 2, vn, i, packet, ni.context.cycles, pid);
			    flit->trace_node.push_back(ni.id);
			    flit->trace_time.push_back(ni.context.cycles);
			    flit->packet->in_net_time = ni.context.cycles;
			    ni.buffer_list[i]->enqueue(flit);
			}
		  }
	      }
	      ni.state[i] = 1;  // wait for VC allocation
	      return true;
	  }
      }

#ifdef SHARED_PRI
      }
      else{  // for URS to use only vc_per_vn without priority (vc_priority_per_vn)
	  base = vn*ni.vc_per_vn;
	  for(int i=base; i<base+ni.vc_per_vn-ni.vc_priority_per_vn; i++){
	    if(ni.state[i] == 0){  // idle
		//packet_id++;  // if success
		packet->mark(PacketEvent::Packetized, ni.context.cycles);
          packet->send_out_time = ni.context.cycles;
		int length = (packet->length-1)/FLIT_LENGTH+1; //round up
		ni.num_flit += length;
		if(length == 1){
		    Flit* flit = Flit::allocate(0, 10, vn, i, packet, ni.context.cycles, pid);
		    ni.buffer_list[i]->enqueue(flit);
		}else{
		    for(int id=0; id<length; id++){
			if(id == 0){
			    Flit* flit = Flit::allocate(id, 0, vn, i, packet, ni.context.cycles, pid);
			    ni.buffer_list[i]->enqueue(flit);
			}else
			  if(id == length-1){
			      Flit* flit = Flit::allocate(id, 1, vn, i, packet, ni.context.cycles, pid);
			      ni.buffer_list[i]->enqueue(flit);
			  }else{
			      Flit* flit = Flit::allocate(id, 2, vn, i, packet, ni.context.cycles, pid);
			      ni.buffer_list[i]->enqueue(flit);
			  }
		    }
		}
		ni.state[i] = 1;  // wait for VC allocation
		return true;
	    }
	  }
      }
#endif
      return false;


  }
}
