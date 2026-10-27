/*
 * NI.cpp
 *
 */

#include "NI.hpp"
#include "NIPacketizer.hpp"
#include "NIPacketAssembler.hpp"
#include <stdexcept>
#include "../parameters.hpp"

NI::NI (int t_id, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth, int* t_NI_num)
    : context(t_vcNetwork->context)
{
  id = t_id;
  vcNetwork = t_vcNetwork;
  vn_num = t_vn_num;
  vc_per_vn = t_vc_per_vn;
  vc_priority_per_vn = t_vc_priority_per_vn;
  NI_num = t_NI_num;
  // output;
  packetBuffer_list.reserve(vn_num);
  tx_peak.resize(vn_num, 0);
  for(int i=0; i<vn_num; i++){
      PacketBuffer* t_packetBuffer = new PacketBuffer(this, i);
      packetBuffer_list.push_back(t_packetBuffer);
  }
  buffer_list.reserve(vn_num*(vc_per_vn+vc_priority_per_vn));

  for(int i=0; i<vn_num*vc_per_vn; i++){
      FlitBuffer * t_flitBuffer = new FlitBuffer(context, i%vc_per_vn, i/vc_per_vn, i, NI_TX_VC_DEPTH_FLITS);
      buffer_list.push_back(t_flitBuffer);
      out_vc.push_back(-1);
      state.push_back(0);
  }

  for(int i=0; i<vn_num*vc_priority_per_vn; i++){
      FlitBuffer * t_flitBuffer = new FlitBuffer(context, i%vc_priority_per_vn+vc_per_vn, i/vc_priority_per_vn, i, NI_TX_VC_DEPTH_FLITS);
      buffer_list.push_back(t_flitBuffer);
      out_vc.push_back(-1);
      state.push_back(0);
  }

  rr_buffer = 0;
  in_depth = t_in_depth;

  // output;
  ROutPort * vc_rOutPort = new ROutPort(context, 0, 1, 1, 0, NI_TX_PORT_DEPTH_FLITS);
  out_port = vc_rOutPort;

  // input;
  Link * link = new Link((RInPort*)NULL);
  RInPort * vcInPort = new RInPort(context, 0, vn_num, vc_per_vn, vc_priority_per_vn, NI_RX_VC_DEPTH_FLITS, this, link);
  vcInPort->rid[0] = t_id/X_NUM;
  vcInPort->rid[1] = t_id%X_NUM;
  link->rInPort = vcInPort;
  in_port = vcInPort;

  count_vc = 0;
  count_switch = 0;
  packet_buffer_out.resize(2);

  starvation = 0;
  rr_priority_record = 0;

  num_flit = 0;

  total_delay = 0;
  total_num = 0;
  total_delay_URS = 0;
  total_num_URS = 0;

  // for new pooling
  NI_cycle_npooling = 0;
}


bool NI::flitize(Packet* packet, int vn) {
    return NIPacketizer::packetize(*this, packet, vn);
}

void NI::packetDequeue(){
  for(int vn=0; vn<vn_num; vn++){
      	// if (packetBuffer_list[vn]->packet_num > NI_TX_FIFO_DEPTH) {
		// 	std::cout << "[WARNING HW] TX FIFO Overflow Node " << id 
		// 	<<"!MAC is injecting faster than the NoC can absorb." << std::endl;
      	// }

        while(packetBuffer_list[vn]->packet_num != 0 && packetBuffer_list[vn]->read()->message.out_cycle < context.cycles){
          	Packet* packet = packetBuffer_list[vn]->read();
          	assert(vn == packet->vnet);

			if(flitize(packet, vn)){
				packetBuffer_list[vn]->dequeue();
				
				if(packet->receive_class == ReceiveClass::RequestsAndData){
					context.ni_statistics.count_s++;
				}
				if(packet->receive_class == ReceiveClass::ResultsAndAcks){
					context.ni_statistics.count_s_resp++;
              	}
          	}
          	else{
              	break; 
          	}
      	}
  	}
}

void NI::vcAllocation(){
  //cout << "#######" << endl;
  // for priority packet (shared VCs) QoS = 1

#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_vc.begin(); iter<priority_vc.end();){
      if(count_vc == STARVATION_LIMIT)
	break;
      int tag = (*iter);
      if(state[tag]==1){
      	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
      	  if (vc != -1){
      	      out_vc[tag] = vc;
      	      state[tag] = 2; // active
      	      count_vc++;
      	      iter = priority_vc.erase(iter);
      	  }else
      	    iter++;
      }else
	iter++;
  }
  count_vc = 0;
#endif

  // for priority packet (individual VCs) QoS = 3
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(state[tag]==1){
	  int vn_rank = (tag-vn_num*vc_per_vn)/vc_priority_per_vn;
  	  int vc = out_port->out_link->rInPort->vc_allocate_priority(vn_rank);  // 2+vn_rank+0 (individual)
  	  if (vc != -1){
  	      out_vc[tag] = vc;
  	      state[tag] = 2; // active
  	  }
      }
  }

  // for normal packet QoS = 0
  for(int i=0; i<vn_num*vc_per_vn; i++){
      int tag = (i+rr_buffer)%(vn_num*vc_per_vn);
      if(state[tag]==1){

#ifdef SHARED_PRI
	  int vc = out_port->out_link->rInPort->vc_allocate_normal(buffer_list[tag]->read());
#else
	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
#endif

	  if (vc != -1){
	      out_vc[tag] = vc;
	      state[tag] = 2; // active
	  }
      }
  }

 // in case of no normal packet
#ifdef SHARED_VC
  for(;iter<priority_vc.end();){
      int tag = (*iter);
      if(state[tag]==1){
      	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
      	  if (vc != -1){
      	      out_vc[tag] = vc;
      	      state[tag] = 2; // active
      	      iter = priority_vc.erase(iter);
      	  }else
      	    iter++;
      }else
	iter++;
  }
#endif
}

void NI::switchArbitration(){

  // for priority packet (shared VCs)

#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_switch.begin(); iter<priority_switch.end();iter++){
      if(count_switch == STARVATION_LIMIT)
	break;
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num!=0){
	  Flit* flit = buffer_list[tag]->read();

	  if(state[tag]==2 && flit->sched_time < context.cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
	      buffer_list[tag]->dequeue();
	      flit->vc = out_vc[tag];

	     // if(out_port->buffer_list[0]->isFull())
	     	    // cout<< "ni switch" << endl;
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(context.cycles);
	      out_port->buffer_list[0]->enqueue(flit);
	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
	      flit->sched_time = context.cycles;
	      count_switch++;
	      if(flit->type == 1 || flit->type == 10){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
  count_switch = 0;
#endif


  // for priority packet (individual VCs)
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      if(starvation == STARVATION_LIMIT){
	  //cout << starvation << ' ' << id << endl;
	  starvation = 0;
          break;
      }
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num!=0){
  	  Flit* flit = buffer_list[tag]->read();
      if(state[tag]==2 && flit->sched_time < context.cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
  	      buffer_list[tag]->dequeue();
  	      flit->vc = out_vc[tag];
  	      // added
//	      flit->trace_node.push_back(NI::id);
//	      flit->trace_time.push_back(cycles);
  	      out_port->buffer_list[0]->enqueue(flit);
  	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
          flit->sched_time = context.cycles;
  	      if(flit->type == 1 || flit->type == 10)
  		state[tag] = 0; //idle
  	      rr_priority_record = (rr_priority_record+1)%(vn_num*vc_priority_per_vn);
  	      starvation++;
  	      return;
          }
      }
  }


  // for normal packet
  for(int i=0; i<vn_num*vc_per_vn; i++){
      int tag = (i+rr_buffer)%(vn_num*vc_per_vn);
      if(buffer_list[tag]->cur_flit_num!=0){
	      Flit* flit = buffer_list[tag]->read();
	      if(state[tag]==2 && flit->sched_time < context.cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
		  buffer_list[tag]->dequeue();
		  flit->vc = out_vc[tag];
		  // added
//		  flit->trace_node.push_back(NI::id);
//		  flit->trace_time.push_back(cycles);
		  out_port->buffer_list[0]->enqueue(flit);
		  out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
		  flit->sched_time = context.cycles;
		  if(flit->type == 1 || flit->type == 10)
		    state[tag] = 0; //idle
		  rr_buffer = (1+rr_buffer)%(vn_num*vc_per_vn);  // rr + 1, the next buffer will have the priority to switch.
		  return;
	      }
      }
  }

  // in case there is no normal packet

#ifdef SHARED_VC
  for(; iter<priority_switch.end();iter++){
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num!=0){
	  Flit* flit = buffer_list[tag]->read();
	  if(state[tag]==2 && flit->sched_time < context.cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
	      buffer_list[tag]->dequeue();
	      flit->vc = out_vc[tag];

	     // if(out_port->buffer_list[0]->isFull())
	     	    // cout<< "ni switch" << endl;
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(context.cycles);
	      out_port->buffer_list[0]->enqueue(flit);
	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
	      flit->sched_time = context.cycles;
	      if(flit->type == 1 || flit->type == 10){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
#endif

  // for priority packet (individual VCs)
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num!=0){
  	  Flit* flit = buffer_list[tag]->read();
      if(state[tag]==2 && flit->sched_time < context.cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
  	      buffer_list[tag]->dequeue();
  	      flit->vc = out_vc[tag];
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(context.cycles);
  	      out_port->buffer_list[0]->enqueue(flit);
  	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
          flit->sched_time = context.cycles;
  	      if(flit->type == 1 || flit->type == 10)
  		state[tag] = 0; //idle
  	      rr_priority_record = (rr_priority_record+1)%(vn_num*vc_priority_per_vn);
  	      return;
          }
      }
  }
}

void NI::dequeue(){
  if(out_port->buffer_list[0]->cur_flit_num != 0 && out_port->buffer_list[0]->read()->sched_time < context.cycles){
      Flit* flit = out_port->buffer_list[0]->dequeue();
	 // if(out_port->out_link->rInPort->buffer_list[flit->vc]->isFull())
	  //   cout<< "ni dequeue" << endl;
      out_port->out_link->rInPort->buffer_list[flit->vc]->enqueue(flit);
      flit->sched_time = context.cycles;

      if(flit->type == 0 || flit->type == 10){
		  VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port->out_link->rInPort->router_owner);
		  assert(vcRouter != NULL);
#ifdef SHARED_VC
		  if(flit->packet->signal->QoS == 1){
			  out_port->out_link->rInPort->priority_vc.push_back(flit->vc);
			  out_port->out_link->rInPort->priority_switch.push_back(flit->vc);
		  }
#endif
		  flit->packet->mark(PacketEvent::HeadInjected, context.cycles);
          int route_result = vcRouter->getRoute(flit);
		  out_port->out_link->rInPort->out_port[flit->vc] = route_result;
		  assert(out_port->out_link->rInPort->state[flit->vc] == 1);
		  out_port->out_link->rInPort->state[flit->vc] = 2; //wait for vc allocation
      }
  }
}

void NI::inputCheck() {
    NIPacketAssembler::poll(*this);
}

void NI::runOneStep(){
  packetDequeue();
  vcAllocation();
  switchArbitration();

  dequeue();
  inputCheck();
}

NI::~NI (){
  FlitBuffer* flitBuffer;
  while(buffer_list.size()!=0){
      flitBuffer = buffer_list.back();
      buffer_list.pop_back();
      delete flitBuffer;
  }
  PacketBuffer* packetBuffer;
  while(packetBuffer_list.size()!=0){
      packetBuffer = packetBuffer_list.back();
      packetBuffer_list.pop_back();
      delete packetBuffer;
  }
  for (auto& queue : packet_buffer_out)
      for (Packet* packet : queue) Packet::release(packet);
  delete in_port;
  delete out_port;
}

bool NI::trySend(Packet* packet) {
    if (!packet || packet->context != &context || packet->vnet < 0 || packet->vnet >= vn_num)
        throw std::invalid_argument("Invalid NI transmit packet");
    if (packet->in_pool || packet->release_requested || packet->submitted)
        throw std::logic_error("NI requires an active packet that has not been submitted");
    if (NI_TX_PACKET_CAPACITY > 0 &&
        txQueued(packet->virtualNetwork()) >= NI_TX_PACKET_CAPACITY) {
        ++tx_rejected_attempts;
        return false;
    }
    packet->submitted = true;
    packet->mark(PacketEvent::Enqueued, context.cycles);
    packetBuffer_list[packet->vnet]->enqueue(packet);
    tx_peak[packet->vnet] = std::max(tx_peak[packet->vnet], txQueued(packet->virtualNetwork()));
    return true;
}
size_t NI::txQueued(VirtualNetwork network) const {
    return packetBuffer_list.at(vnIndex(network))->packet_num;
}
Packet* NI::ReceiveCursor::peek() const {
    const auto& queue = owner_.packet_buffer_out.at(queueIndex(category_));
    return index_ < queue.size() ? queue[index_] : nullptr;
}
void NI::ReceiveCursor::skip() {
    if (peek()) ++index_;
}
Packet* NI::ReceiveCursor::take() {
    auto& queue = owner_.packet_buffer_out.at(queueIndex(category_));
    Packet* packet = peek();
    if (packet) queue.erase(queue.begin() + index_);
    return packet;
}
