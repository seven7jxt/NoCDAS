/*
 * VCRouter.cpp
 *
 */

#include "VCRouter.hpp"
#include "../ComputeTiming.hpp"
#include "../parameters.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

static_assert(ROUTER_MACS_PER_CYCLE > 0, "Router MAC throughput must be positive");
static_assert(ROUTER_SRAM_WIDTH > 0 &&
              ROUTER_SRAM_WIDTH % (DATA_BYTES * 8) == 0,
              "Router SRAM interface must hold a whole number of data elements");

VCRouter::VCRouter(int* t_id, int in_out_port_num, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth, std::unique_ptr<ComputeUnit> unit)
    : context(t_vcNetwork->context), mfu(std::move(unit))
{
    id[0] = t_id[0];  
    id[1] = t_id[1];  

    vcNetwork = t_vcNetwork;

    in_port_list.reserve(in_out_port_num);
    out_port_list.reserve(in_out_port_num);
    
    local_weights.clear();
    local_kv_cache.clear();

    mfu_waiting_for_kv = false;
    kv_retry_until = 0;

    int total_vcs_per_port = t_vn_num * (t_vc_per_vn + t_vc_priority_per_vn);
    
    // [5 ports: up, right, down, left, local] x [N VCs per port]
    vc_compute_state.resize(in_out_port_num, std::vector<ComputeVCState>(total_vcs_per_port));

    for (int i=0; i< in_out_port_num; i++){
        Link * link = new Link((RInPort*)NULL);
        RInPort * t_rInPort = new RInPort(context, i, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_in_depth, this, link);
        t_rInPort->rid[0] = t_id[0];
        t_rInPort->rid[1] = t_id[1];

        link->rInPort = t_rInPort;
        in_port_list.push_back(t_rInPort);

    #ifdef outPortNoInfinite
        ROutPort * t_rOutPort = new ROutPort(context, i, 1, 1, 0, i == 4 ? 2 : 1);
    #else
        ROutPort * t_rOutPort = new ROutPort(context, i, 1, 1, 0, ROUTER_LARGE_OUTPUT_DEPTH_FLITS);
    #endif
        out_port_list.push_back(t_rOutPort);
    }
    rr_port = 0;
    rr_out_port = 0;
    port_num = in_out_port_num;
    port_total_utilization = 0;
    port_utilization_innet = 0;
    
    current_sram_usage = 0;
    local_weights.reserve(ROUTER_SRAM_LIMIT);
    local_kv_cache.reserve(ROUTER_SRAM_LIMIT);
    
    int total_vcs = t_vn_num * (t_vc_per_vn + t_vc_priority_per_vn);
    vc_compute_state.resize(in_out_port_num, std::vector<ComputeVCState>(total_vcs));

    local_kv_cache.reserve(ROUTER_SRAM_LIMIT);
    local_weights.reserve(ROUTER_SRAM_LIMIT);
    kv_token_count = 0;
}

int VCRouter::getRoute(Flit* t_flit){
      // cNoC: Source Routing (Wormhole Safe)
      if (t_flit->packet->message.kind == MessageKind::Distribution || t_flit->packet->message.kind == MessageKind::Compute) {
          int path_idx = t_flit->packet->current_path_index;
          auto& path = t_flit->packet->message.routing_path;

          if (path_idx < path.size() && path[path_idx] == (id[0] * X_NUM + id[1])) {
              path_idx++;
              t_flit->packet->current_path_index = path_idx;
          }
          
          if (path_idx < path.size()) {
              int next_router = path[path_idx];
              int target_x = next_router / X_NUM;
              int target_y = next_router % X_NUM;
              if (target_y < id[1]) return 3; // turn left
              if (target_y > id[1]) return 1; // turn right
              if (target_x < id[0]) return 0; // turn up
              if (target_x > id[0]) return 2; // turn down
          }
      }

      int x = t_flit->packet->destination[0];
      int y = t_flit->packet->destination[1];
      int z = t_flit->packet->destination[2];
      
      if(y < id[1]) return 3; 
      else if(y > id[1]) return 1; 
      else { 
          if(x < id[0]) return 0; 
          else if(x > id[0]) return 2; 
          else return (z+4); 
      }
      return -1;
}

void VCRouter::processDistributionPacket(Flit* t_flit) {
    sram_layer = t_flit->packet->message.layer_id;
    int flat_router_id = id[0] * X_NUM + id[1]; 

    if (t_flit->computed_routers.test(flat_router_id)) {
        return;
    }
    
    t_flit->computed_routers.set(flat_router_id);

    int payload_size = t_flit->get_payload_size();
    int opcode = t_flit->packet->message.compute_op;

    if (opcode == ATTENTION) { // Opcode for KV Cache loading
        /*
         * STREAMING LLM IMPLEMENTATION (Attention Sinks + Rolling KV Cache) -> https://arxiv.org/abs/2309.17453
         * The paper demonstrates that a surprisingly large amount of attention score (often >50%) 
         * is allocated to the very first tokens ("Attention Sinks"). By keeping the KV cache of 
         * just the first 4 tokens fixed, and using a sliding window for the most recent tokens 
         * (Rolling KV Cache), a Transformer can generalize to infinite-length sequences without 
         * fine-tuning, keeping stability and perplexity unchanged.
         * * [HARDWARE MAPPING]
         * In our cNoC architecture, the "Rolling KV Cache" is physically implemented as a Ring Buffer 
         * inside the Router's SRAM. We lock the first 4 tokens (SINK_TOKENS) and circularly overwrite 
         * the remaining space up to the ROUTER_SRAM_LIMIT to prevent hardware memory overflow.
         * ====================================================================================
         */
        const int SINK_TOKENS = 4;
        
        // A KV loading packet carries [K, V] for a token, so the size is twice the head_dim
    //     int floats_per_token = payload_size; 
    //     int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;

    //     if (local_kv_cache.size() + payload_size <= ROUTER_SRAM_LIMIT) {
    //         // PREFILL phase: There is still space, insert normally
    //         for (int i = 0; i < payload_size; i++) {
    //             local_kv_cache.push_back(t_flit->get_data(i));
    //         }
    //         current_sram_usage += payload_size;
    //     } 
    //     else {
    //         // SRAM FULL: Eviction is triggered (Ring Buffer)
    //         // The index rotates discarding intermediate tokens but SAVING the Sinks (0, 1, 2, 3)
    //         int ring_index = SINK_TOKENS + ((kv_token_count - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
    //         int offset = ring_index * floats_per_token;
            
    //         // Overwrite old data in O(1), zero memory reallocations
    //         for (int i = 0; i < payload_size; i++) {
    //             local_kv_cache[offset + i] = t_flit->get_data(i);
    //         }
    //     }
    //     kv_token_count++;
    // } 
        int floats_per_token = t_flit->packet->message.data_length; 
        int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;

        if (kv_token_count < max_tokens_in_sram) {
            // Prefill phase
            int write_pos = (kv_token_count * floats_per_token) + t_flit->global_data_offset;
            
            for (int i = 0; i < payload_size; i++) {
                writeKV(write_pos + i, t_flit->get_data(i));
            }
        } 
        else {
            // Eviction phase (Ring Buffer)
            int ring_index;
            if (max_tokens_in_sram > SINK_TOKENS) {
                ring_index = SINK_TOKENS + ((kv_token_count - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
            } else {
                ring_index = std::max(0, max_tokens_in_sram - 1);
            }
            
            int base_offset = ring_index * floats_per_token;

            // A distribution packet is split into several flits.  Report one
            // event per logical token, at the tail, rather than one event per
            // flit.  sequence_id is the token position in the current layer.
            if (t_flit->type == 1 || t_flit->type == 10) {
                const int layer_id = t_flit->packet->message.layer_id;
                const int logical_token = t_flit->packet->message.sequence_id;
                std::cerr << "KV_EVICTION layer=" << layer_id
                          << " router=(" << id[0] << "," << id[1] << ")"
                          << " token=" << logical_token
                          << " local-token=" << kv_token_count
                          << " slot=" << ring_index
                          << " capacity-tokens=" << max_tokens_in_sram
                          << " sink-tokens=" << SINK_TOKENS << std::endl;
                vcNetwork->recordKVEviction(layer_id);
            }
            
            for (int i = 0; i < payload_size; i++) {
                // Secure overwrite (writeKV will understand that index < size and won't allocate more SRAM)
                writeKV(base_offset + t_flit->global_data_offset + i, t_flit->get_data(i));
            }
        }
        
        if (t_flit->type == 1 || t_flit->type == 10) {
            kv_token_count++;
        }
    } else {
        if (t_flit->id == 0) { 
             this->clearWeights();
        }
        
        int payload_size = t_flit->get_payload_size();
        for (int i = 0; i < payload_size; i++) {
            storeWeight(t_flit->get_data(i));
        }
    }
}

void VCRouter::vcRequest(){  
  for(int i=0; i<port_num; i++){ 
      in_port_list[(i+rr_port)%port_num]->vc_request();
  }
}

void VCRouter::getSwitch(){
  for(int i=0; i<port_num; i++){ 
      in_port_list[(i+rr_port)%port_num]->getSwitch();
  }
  rr_port = (rr_port+1)%port_num;
}

Cycle VCRouter::reserveMFU(Cycle duration) {
    [[maybe_unused]] const bool accepted = mfu->tryIssue({duration}, context.cycles);
    assert(accepted);
    return mfu->scheduled().ready_at;
}

void VCRouter::outPortDequeue(){
    // This is a per-attempt diagnostic.  The retry interval itself is kept in
    // kv_retry_until so a blocked port can still be classified correctly.
    mfu_waiting_for_kv = false;
    for(int count = 0; count < port_num; count++){ 
        int i = (rr_out_port + count) % port_num;

        if(out_port_list[i]->buffer_list[0]->cur_flit_num != 0 && out_port_list[i]->buffer_list[0]->read()->sched_time < context.cycles){
            Flit* flit = out_port_list[i]->buffer_list[0]->read();

            if (flit->packet->message.kind == MessageKind::Distribution || flit->packet->message.kind == MessageKind::Compute) {
                // Retire before each candidate, preserving same-cycle reuse
                // for a zero-duration reservation as well as normal completion.
                ComputeCompletion completion;
                mfu->tryComplete(context.cycles, completion);
                if (mfu->busy()) {
                    // A KV retry occupies the router-wide MFU, but other
                    // eligible flits blocked by that same hold are waiting
                    // for the MFU, not for their own KV data.
                    const bool is_attention =
                        flit->packet->message.kind == MessageKind::Compute &&
                        flit->packet->message.compute_op == ATTENTION;
                    const bool waiting_for_kv = is_attention &&
                                                context.cycles < kv_retry_until;
                    mfu_waiting_for_kv = waiting_for_kv;
                    vcNetwork->recordRouterWait(id[0] * X_NUM + id[1], flit->packet->message.layer_id,
                                                waiting_for_kv);
                    continue;
                }
            }
            
            if (flit->packet->message.kind == MessageKind::Compute &&
                flit->packet->message.compute_op == ATTENTION &&
                (flit->type == 1 || flit->type == 10)) {
                int this_router_id = id[0] * X_NUM + id[1];
                int N = flit->packet->message.routing_path.size() - 1;
                int router_idx = -1;
                for (int r = 0; r < N; r++) {
                    if (flit->packet->message.routing_path[r] == this_router_id) {
                        router_idx = r; break;
                    }
                }
                if (router_idx != -1) {
                    int current_query_y = flit->packet->message.sequence_id;
                    int expected_local_tokens = 0;
                    for (int y = 0; y <= current_query_y; y++) {
                        if ((y % N) == router_idx) expected_local_tokens++;
                    }
                    
                    if (kv_token_count < expected_local_tokens) {
                        vcNetwork->recordRouterWait(id[0] * X_NUM + id[1], flit->packet->message.layer_id, true);
                        mfu_waiting_for_kv = true;
                        // A retry holds the same resource without executing a kernel.
                        kv_retry_until = reserveMFU(2);
                        continue;
                    }
                }
            }

            flit = out_port_list[i]->buffer_list[0]->dequeue();

            int compute_delay = 0;

            if (flit->packet->message.kind == MessageKind::Distribution) {
                int this_router = id[0] * X_NUM + id[1];
                
                // Snooping: The router intercepts the in-transit data and populates the SRAM
                bool is_target = false;
                for (int r : flit->packet->message.routing_path) {
                    if (r == this_router) is_target = true;
                }
                if (flit->packet->destination[0] == id[0] && flit->packet->destination[1] == id[1]) {
                    is_target = true;
                }

                if (is_target) {
                    if (!flit->computed_routers.test(this_router)) {
                        compute_delay = LegacyComputeTiming::routerDistribution(flit->get_payload_size(), context.compute.router);
                    }
                    processDistributionPacket(flit);
                }
                
                compute_delay = std::max(1, compute_delay);
            }
            else if (flit->packet->message.kind == MessageKind::Compute) {
                const int matmul_macs = computeInTransit(flit, i);
                
                int opcode = flit->packet->message.compute_op;
                
                if (opcode == ATTENTION) {
                    // Normalize at the same tail event as before; timing has no data side effects.
                    const auto work = finishAttention(flit);
                    compute_delay = LegacyComputeTiming::routerAttention(work, context.compute.router);
                } else {
                    compute_delay = LegacyComputeTiming::routerOperator(opcode, matmul_macs, context.compute.router);
                }
            }

            const Cycle ready =
                flit->packet->message.kind == MessageKind::Distribution ||
                flit->packet->message.kind == MessageKind::Compute
                ? reserveMFU(compute_delay) : context.cycles;

            out_port_list[i]->out_link->rInPort->buffer_list[flit->vc]->enqueue(flit);

            // Count traffic crossing a physical router-to-router link. Local
            // NI transfers use ports >= 4 and are intentionally excluded.
            if (i < 4) {
                vcNetwork->recordFlitHop(flit->packet->message.layer_id);
            }
            
            flit->sched_time = ready + LINK_TIME - 1;

            flit->trace_node.push_back(out_port_list[i]->out_link->rInPort->rid[0]*X_NUM + out_port_list[i]->out_link->rInPort->rid[1]);
            flit->trace_time.push_back(context.cycles + LINK_TIME - 1);

            port_total_utilization++;
            if (i<=3) port_utilization_innet++;

            if(flit->type == 0 || flit->type == 10){
                VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port_list[i]->out_link->rInPort->router_owner);
                if (vcRouter != NULL){
#ifdef SHARED_VC 
                    if(flit->packet->message.QoS == 1){
                        out_port_list[i]->out_link->rInPort->priority_vc.push_back(flit->vc);
                        out_port_list[i]->out_link->rInPort->priority_switch.push_back(flit->vc);
                    }
#endif
                    int route_result = vcRouter->getRoute(flit);
                    out_port_list[i]->out_link->rInPort->out_port[flit->vc] = route_result;
                    assert(out_port_list[i]->out_link->rInPort->state[flit->vc] == 1);
                }
                out_port_list[i]->out_link->rInPort->state[flit->vc] = 2; 
            }
        }
    }
  rr_out_port = (rr_out_port + 1) % port_num;
}

void VCRouter::runOneStep(){
    vcRequest();
    getSwitch();
    outPortDequeue();
    vcNetwork->sampleRouterOccupancy(id[0] * X_NUM + id[1], mfu->scheduled().ready_at, kv_retry_until);
}

bool VCRouter::allocateSRAM(int num_floats) {
    if (current_sram_usage + num_floats > ROUTER_SRAM_LIMIT) {
        std::cerr << "\n[HARDWARE EXCEPTION] Router (" << id[0] << "," << id[1] 
                  << ") SRAM Overflow! Limit: " << ROUTER_SRAM_LIMIT 
                  << " Used: " << current_sram_usage 
                  << " Requested: " << num_floats << std::endl;
        return false; 
    }
    current_sram_usage += num_floats;
    return true;
}

void VCRouter::recordSramUsage() {
    sram_stats.observe(static_cast<std::uint64_t>(local_weights.size()) * DATA_BYTES, 0,
                       static_cast<std::uint64_t>(local_kv_cache.size()) * DATA_BYTES,
                       context.cycles, sram_layer);
}

void VCRouter::storeWeight(float weight_value) {
    bool can_allocate = allocateSRAM(1);
    if (!can_allocate) {
        std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to store weight in router (" 
                  << id[0] << "," << id[1] << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    local_weights.push_back(weight_value);
    recordSramUsage();
}

void VCRouter::storeKV(float kv_value) {
    bool can_allocate = allocateSRAM(1);
    if (!can_allocate) {
        std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to store KV cache in router (" 
                  << id[0] << "," << id[1] << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    local_kv_cache.push_back(kv_value);
    recordSramUsage();
}

void VCRouter::writeKV(int index, float kv_value) {
    if (index >= local_kv_cache.size()) {
        int needed_expansion = (index + 1) - local_kv_cache.size();
        
        bool can_allocate = allocateSRAM(needed_expansion);
        if (!can_allocate) {
            std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to expand KV cache in router (" 
                      << id[0] << "," << id[1] << ")." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        local_kv_cache.resize(index + 1, 0.0f);
        recordSramUsage();
    }
    local_kv_cache[index] = kv_value;
}

void VCRouter::clearSRAM() {
    local_weights.clear();
    local_kv_cache.clear();
    current_sram_usage = 0;
    kv_token_count = 0;
    kv_retry_until = 0;

    kv_token_count = 0;
    assigned_tasks.clear();
}

void VCRouter::clearWeights() {
    int weight_size = local_weights.size();
    local_weights.clear();
    current_sram_usage -= weight_size; 
}

VCRouter::~VCRouter ()
{
  RInPort* inPort;
    while(in_port_list.size()!=0){
        inPort = in_port_list.back();
        in_port_list.pop_back();
        delete inPort;
    }

    ROutPort* outPort;
        while(out_port_list.size()!=0){
            outPort = out_port_list.back();
            out_port_list.pop_back();
            delete outPort;
    }
}
