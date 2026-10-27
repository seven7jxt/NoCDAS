#include "NIPacketAssembler.hpp"
#include "NI.hpp"
#include <stdexcept>

void NIPacketAssembler::poll(NI& ni) {
  for(int i=0; i<ni.vn_num*(ni.vc_per_vn + ni.vc_priority_per_vn); i++){
      ///< if the in port buffer contains flit
      if(ni.in_port->buffer_list[i]->cur_flit_num != 0){
          assert(ni.in_port->state[i] == 2);
          Flit* flit = ni.in_port->buffer_list[i]->readLast(); ///> reference to the last flit in buffer
          int length = (flit->packet->length-1)/FLIT_LENGTH+1;

          /// normal check
          if(flit->sched_time < ni.context.cycles && (flit->type == 1 || flit->type == 10)){  ///< tail or head tail flit
              Packet* packet = flit->packet;
              packet->mark(PacketEvent::TailArrived, ni.context.cycles);

              // --- RTL FIFO BACKPRESSURE LOGIC ---
              // Determiniamo la FIFO di destinazione (0 per Request, 1 per Response/cNoC)
              int target_fifo = (packet->receive_class == ReceiveClass::RequestsAndData) ? 0 : 1; 
              
              // Se la FIFO di destinazione è piena, applichiamo la contropressione
              if (ni.packet_buffer_out[target_fifo].size() >= NI_RX_FIFO_DEPTH) {
                  // Ignoriamo questo flit per il ciclo corrente. 
                  // Non arrivando al while() in fondo, i flit non vengono estratti.
                  // I crediti non vengono restituiti, fermando il router a monte.
                  continue; 
              }
              // -----------------------------------

              if (ni.in_port->buffer_list[i]->cur_flit_num != length)
                  throw std::logic_error("Incomplete packet at NI assembly");
              int expected_id = 0;
              for (const auto* part : ni.in_port->buffer_list[i]->flit_queue) {
                  if (part->packet != packet || part->id != expected_id++)
                      throw std::logic_error("Mixed or reordered flits at NI assembly");
              }
              packet->mark(PacketEvent::Delivered, ni.context.cycles);
              ni.in_port->state[i] = 0; //idle; 2->0; no need for vc allocation
              
#ifndef newpooling
              packet->message.out_cycle = ni.context.cycles + LINK_TIME - 1;   // 1 cycle delay from NI_network to masterNI/slaveNI
              ni.NI_cycle_npooling = ni.context.cycles;
#endif
#ifdef newpooling
              if (packet->message.penable > 0)
              {
                  if (ni.NI_cycle_npooling < ni.context.cycles + 4)
                  {
                      ni.NI_cycle_npooling = ni.context.cycles + POOLING_DELAY;
                      packet->message.out_cycle = ni.NI_cycle_npooling + LINK_TIME - 1; // pipeline the pooling unit
                  }
                  else
                  {
                      ni.NI_cycle_npooling = ni.NI_cycle_npooling + POOLING_DELAY;
                      packet->message.out_cycle = ni.NI_cycle_npooling + LINK_TIME - 1; // add pooling unit delay
                  }
                  // cout << cycles << " activate pooling at cycle " << packet->message.out_cycle << endl;
              }
              else
              {
                  packet->message.out_cycle = ni.context.cycles + LINK_TIME - 1;   // 1 cycle delay from NI_network to masterNI/slaveNI
                  ni.NI_cycle_npooling = ni.context.cycles;
              }
#endif
              if(packet->receive_class == ReceiveClass::RequestsAndData){ // request type 0
                  ni.packet_buffer_out[0].push_back(packet);
                  ni.context.ni_statistics.count_r++;
                  //cout << "NI received " << count_r << endl;
              }
              else{ // response type 1
                  ni.context.ni_statistics.count_r_resp++;
                  //cout <<  "NI received response " << count_r_resp << endl;
                  ni.packet_buffer_out[1].push_back(packet);
              }


              if(packet->message.QoS == 3 || packet->message.QoS == 1){
              ni.total_delay += ni.context.cycles - packet->send_out_time;
              ni.total_num++;
              }else{
              ni.total_delay_URS += ni.context.cycles - packet->send_out_time;
              ni.total_num_URS++;
              }


              std::int64_t delay = static_cast<std::int64_t>(ni.context.cycles - packet->send_out_time) - (abs(packet->message.NI_id/X_NUM-packet->destination[0])+abs(packet->message.NI_id%X_NUM-packet->destination[1])+1)*3 - 2 - ((packet->length-1)/FLIT_LENGTH+1);

            // cNoC packets (type 4/5) have additional latency from in-transit computation and 
            // do not respect the lower bound of pure XY routing: exclude them from the assert.
            if (packet->message.kind != MessageKind::Distribution && packet->message.kind != MessageKind::Compute) {
                assert(delay >= 0);
            }
            // A source-routed compute packet can violate the XY lower bound;
            // never use its negative excess delay as a histogram index.
            delay = std::max<std::int64_t>(0, delay);

#ifdef STD_LATENCY
          if(packet->signal->test_tag == 1){
#else
          if(packet->message.QoS==3 || packet->message.QoS==1){
#endif

          ni.context.ni_statistics.worst_LCS = (ni.context.ni_statistics.worst_LCS<delay)?delay:ni.context.ni_statistics.worst_LCS;
          if(delay < DISTRIBUTION_NUM-1)
            ni.context.ni_statistics.LCS_delay_distribution[delay]++;
          else
            ni.context.ni_statistics.LCS_delay_distribution[DISTRIBUTION_NUM-1]++;}
          else{
          ni.context.ni_statistics.worst_URS = (ni.context.ni_statistics.worst_URS<delay)?delay:ni.context.ni_statistics.worst_URS;
          if(delay < DISTRIBUTION_NUM-1)
            ni.context.ni_statistics.URS_delay_distribution[delay]++;
          else
            ni.context.ni_statistics.URS_delay_distribution[DISTRIBUTION_NUM-1]++;
          }


          //std::cout<<"Receive packet at cycles: "<< cycles << ", at NI id: " << id << ". Packet length:" <<packet->length <<". slave id:" << packet->slave_id << ". sequence id:" << packet->sequence_id<<std::endl;

              // ideal_time = (hop+1)*2+flit_num; hop: the routers the packet get through
          	while(ni.in_port->buffer_list[i]->cur_flit_num != 0){
				Flit* flit = ni.in_port->buffer_list[i]->dequeue();
				length = (flit->packet->length-1)/FLIT_LENGTH+1;

				Flit::release(flit);
          }
      }
      }
  }
}
