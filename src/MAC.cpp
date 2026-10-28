/*
 * MAC.cpp
 *
 */

#include "MAC.hpp"
#include "ComputeTiming.hpp"
#include "ComputeKernels.hpp"

void MAC::recordSramUsage() {
    sram_stats.observe(static_cast<std::uint64_t>(weight.size()) * DATA_BYTES,
                       static_cast<std::uint64_t>(infeature.size()) * DATA_BYTES,
                       static_cast<std::uint64_t>(kv_cache.size()) * DATA_BYTES,
                       context.cycles, net->c_layer);
    // Attention scores are unquantized intermediates; report separately from data SRAM.
    score_cache_peak_bytes = std::max(score_cache_peak_bytes,
        static_cast<std::uint64_t>(cached_attention_scores.size()) * 4);
}

static_assert(PE_GATE_LATENCY > 0, "Gate timing must be positive");
static_assert(PE_GATE_BATCH_PAIRS > 0 && MAC_INPUT_SRAM_LIMIT >= 2,
              "Activation SRAM must hold a gate/up pair");
static_assert(PE_ADD_BATCH_PAIRS > 0 && PE_NUM_OP > 0 && ADD_LATENCY > 0,
              "ADD batch size, throughput and latency must be positive");

MAC::MAC (int t_id, MACnet* t_net, int t_NI_id, std::unique_ptr<ComputeUnit> unit)
    : context(t_net->context), compute_unit(std::move(unit))
{
    id = t_id;
    net = t_net;
    NI_id = t_NI_id;
    weight.clear();
    infeature.clear();
    inbuffer.clear();
    ch_size = 0;
    m_size = 0;
    fn = -1;
    tmpch = -1;
    tmpm = 0;
    request = -1;
    tmp_request = -1;
    use_matmul_tiling = false;
    matmul_activation_valid = false;
    matmul_activation_row = -1;
    matmul_requested_row = -1;
    matmul_include_activation = true;
    matmul_tile_start_task = -1;
    matmul_tile_count = 0;
    matmul_tile_weight_size = 0;
    pending_acks = 0;
    received_acks = 0;
    matmul_tile_compute_cycles = 0;
    cached_score_row = -1;
    cached_score_head = -1;

    outfeature = 0.0;
    nextMAC = NULL;
    pecycle = 0;
    selfstatus = 0;
    send = 0;
    m_count = 0;

    // for new pooling
    npoolflag = 0;
    n_tmpch = 0;
    n_tmpm.clear();

    // for Transformer
    causal_mask = 0;
    local_sram_usage = 0;

    data_wait_active = false;
    data_wait_memory_ready = false;
    data_wait_last_cycle = 0;
    data_wait_memory_start = 0;
    data_wait_memory_ready_cycle = 0;
    data_wait_layer = -1;

    current_chunk = 0;
    total_chunks = 1;
    psum_accumulator = 0.0f;

    // find dest id
    int xid = NI_id / X_NUM;
    int yid = NI_id % X_NUM;
    // MC nodes
#ifdef MemNode4
    if (xid <= 3)
    {
        dest_mem_id = dest_list[(yid/4)];
    }
    else
    {
        dest_mem_id = dest_list[(yid/4) + 2];
    }
#elif defined MemNode2
    dest_mem_id = dest_list[(yid/2)];
#elif defined MemNode8
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else {
        dest_mem_id = dest_list[(yid/2) + 4];
    }
#elif defined MemNode8edge
    if (yid <= 3) {
        dest_mem_id = dest_list[(xid/2)*2];     // left 0 2 4 6
    } else {
        dest_mem_id = dest_list[(xid/2)*2 + 1]; // right 1 3 5 7
    }
#elif defined MemNode18                         // 12*12
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 6];
    } else {
        dest_mem_id = dest_list[(yid/2) + 12];
    }
#elif defined MemNode32                         // 16*16
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 8];
    } else if (xid <= 11 && xid > 7 ) {
        dest_mem_id = dest_list[(yid/2) + 16];
    } else {
        dest_mem_id = dest_list[(yid/2) + 24];
    }
#elif defined MemNode5                          // 6*6
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else {
        dest_mem_id = dest_list[(yid/3) + 3];
    }
#elif defined MemNode13                         // 10*10
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 5];
    } else {
        dest_mem_id = dest_list[(yid/4) + 10];
    }
#endif
    routing_table.clear();
}


bool MAC::inject (MessageKind kind, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src)
{
    if (kind == MessageKind::DataRequest && !data_wait_active) {
        data_wait_active = true;
        data_wait_memory_ready = false;
        data_wait_last_cycle = context.cycles;
        data_wait_memory_start = context.cycles;
        data_wait_memory_ready_cycle = context.cycles;
        data_wait_layer = net->c_layer;
    }

    // MC responses are scheduled after the MC has assigned its pecycle.  Keep
    // the ready interval on the requesting PE, rather than on the shared MC,
    // because several PEs may have outstanding requests simultaneously.
    if (kind == MessageKind::DataResponse && mac_src >= 0 && mac_src < static_cast<int>(net->MAC_list.size())) {
        MAC* requester = net->MAC_list[mac_src];
        if (requester->data_wait_active) {
            requester->data_wait_memory_ready = true;
            requester->data_wait_memory_start = context.cycles;
            requester->data_wait_memory_ready_cycle =
                std::max<std::uint64_t>(context.cycles, this->pecycle);
        }
    }

    Message msg{};
    msg.NI_id = NI_id;
    msg.mac_id = mac_src;                       //MAC
    msg.data_length = data_length;
    int selector = rand()%90;
#ifdef LCS_URS_TRAFFIC
    if(selector >= 45) 
        msg.QoS = 3;
    else 
        msg.QoS = 0;
#endif
#ifdef SHARED_VC
    if(msg.QoS == 3) 
        msg.QoS = 1;
#endif
    msg.QoS = 0;

    msg.data.assign(1, t_output);
    msg.data.push_back(tmpch);
    msg.data.push_back(tmpm);

    msg.penable = this->npoolflag;

    msg.destination = d_id;
    msg.out_cycle = pecycle;
    msg.sequence_id = 0;
    msg.signal_id = p_id;
    msg.layer_id = net->c_layer;
    msg.slave_id = d_id;                       //NI
    msg.source_id = NI_id;                     // NI
    msg.kind = kind;                           // 0 1 2 3

    // Packet* packet = new Packet(std::move(msg), X_NUM, t_NI->NI_num);
    Packet* packet = Packet::allocate(context, std::move(msg), X_NUM, t_NI->NI_num);
    packet->send_out_time = pecycle;
    packet->in_net_time = pecycle;

    submitPacket(packet);

    return true;
}

void MAC::submitPacket(Packet* packet) {
    if (!pending_tx.empty() || !net->vcNetwork->NI_list[NI_id]->trySend(packet))
        pending_tx.push_back(packet);
}

void MAC::retryPendingTx() {
    while (!pending_tx.empty()) {
        if (!net->vcNetwork->NI_list[NI_id]->trySend(pending_tx.front())) return;
        pending_tx.pop_front();
    }
}

void MAC::accountDataWait()
{
    if (!data_wait_active || context.cycles <= data_wait_last_cycle) return;

    const std::uint64_t begin = data_wait_last_cycle;
    const std::uint64_t end = context.cycles;
    std::uint64_t memory_cycles = 0;
    if (data_wait_memory_ready) {
        const std::uint64_t memory_begin = data_wait_memory_start;
        const std::uint64_t memory_end = data_wait_memory_ready_cycle;
        const std::uint64_t overlap_begin = std::max(begin, memory_begin);
        const std::uint64_t overlap_end = std::min(end, memory_end);
        if (overlap_end > overlap_begin) {
            memory_cycles = overlap_end - overlap_begin;
        }
    }

    const std::uint64_t elapsed = end - begin;
    net->vcNetwork->recordPEWait(id, data_wait_layer, elapsed - memory_cycles,
                                  memory_cycles);
    data_wait_last_cycle = end;
}

void MAC::finishDataWait()
{
    if (!data_wait_active) return;
    accountDataWait();
    data_wait_active = false;
}


void MAC::scheduleCompute(Cycle duration)
{
    [[maybe_unused]] const bool accepted = compute_unit->tryIssue({duration}, context.cycles);
    assert(accepted);
    pecycle = compute_unit->scheduled().ready_at;
}

void MAC::runOneStep()
{
    // Completion is available at ready_at. The existing strict pecycle < now
    // gate still controls PE transitions; pecycle also carries non-compute waits.
    ComputeCompletion completion;
    if (compute_unit) compute_unit->tryComplete(context.cycles, completion);
    // Sample before scheduler gating so a PE waiting for a response is still
    // charged while its local pecycle is in the future.
    accountDataWait();
    retryPendingTx();
    if (hasPendingTx()) {
        // Earlier results of a batch may already return while later results
        // are still blocked. Allow their ACKs without completing the task.
        if (selfstatus == 4 && pecycle < context.cycles) send = 1;
        return;
    }

    // output stationary (neuron based calculation)
    if (pecycle < context.cycles){
        // initial idle state

        if(selfstatus == 0)
        {
            if(routing_table.size()==0)
            {
                selfstatus = 0;
                pecycle = context.cycles;
            }
            else
            {
                pecycle = context.cycles;
                selfstatus = 1;
            }
        }
        // request data state
        else if(selfstatus == 1)
        {
            if (current_chunk == 0) {
                request = routing_table.front();
                tmp_request = request;
                routing_table.pop_front();
                if (use_elementwise_batch) {
                    elementwise_tasks.assign(1, request);
                    const size_t capacity = std::min(net->o_fn == ADD ? PE_ADD_BATCH_PAIRS : PE_GATE_BATCH_PAIRS,
                                                     MAC_INPUT_SRAM_LIMIT / 2);
                    while (!routing_table.empty() && elementwise_tasks.size() < capacity) {
                        elementwise_tasks.push_back(routing_table.front());
                        routing_table.pop_front();
                    }
                }
                if (use_matmul_tiling) {
                    matmul_tile_start_task = request;
                    matmul_requested_row = request / net->o_x;
                    const int row_size = net->weight_table.empty()
                        ? 0 : static_cast<int>(net->weight_table[0].size());
                    assert(row_size > 0 && "Missing weight row for MatMul/Linear tile");
                    matmul_tile_weight_size = row_size;
                    matmul_tile_count = std::min(
                        MAC_WEIGHT_SRAM_LIMIT / row_size,
                        net->o_x - (request % net->o_x));
                    assert(matmul_tile_count > 0 && "Weight SRAM cannot hold one weight row");
                    matmul_include_activation =
                        !matmul_activation_valid || matmul_activation_row != matmul_requested_row;
                }
            }
            inject(MessageKind::DataRequest, dest_mem_id, 1, request, net->vcNetwork->NI_list[NI_id], context.packet_id_base + request, id);
            selfstatus = 2;
            pecycle = context.cycles;
        }
        else if(selfstatus == 2)
        {
            if(request >= 0) {pecycle = context.cycles; selfstatus = 2; return;}
            assert((inbuffer.size() >= 4) && "Inbuffer not correct after request is set to 0");

            // inbuffer: [fn]
            fn = inbuffer[0]; 
#ifdef newpooling
            if(this->npoolflag == 1 && this->n_tmpch == -1)
            {
                assert((fn == 10) && "Inbuffer not correct when merged with pooling");
                if(this->routing_table.size()==0)
                {
                    this->selfstatus = 5;
                    this->send = 3;
                }
                else
                {
                    this->selfstatus = 0;               // back to initial state
                    this->send = 0;
                }
                // cout << "from mac " << this->id << " abandoned at cycles " << cycles << " " << selfstatus << endl;
                this->weight.clear();
                this->infeature.clear();
                this->inbuffer.clear();
                this->outfeature = 0.0;
                this->npoolflag = 0;
                this->n_tmpch = -1;
                this->n_tmpm.clear();
                this->pecycle = context.cycles + 1; //cycles + 1
                return;
            }
#endif
            if (fn >=0 && fn <=3){                     // Conv [fn] [ch size] [map size] [i] [w + b]
                ch_size = inbuffer[1];
                m_size = inbuffer[2];
                infeature.assign(inbuffer.begin() + 3, inbuffer.begin() + 3 + ch_size * m_size); //input
                weight.assign(inbuffer.begin() + 3 + ch_size * m_size, inbuffer.end()); // w matrix + b (ch_size * m_size + 1)
                assert((weight.size() == ch_size * m_size + 1) && "Weight not correct after request (Conv)");
            }
            else if (fn >= 4 && fn <= 7)              // fc [fn] [map size] [i] [w + b]
            {
                ch_size = 1;
                m_size = inbuffer[1];
                if (use_matmul_tiling && inbuffer[1] < 0) {
                    // Tiled FC response:
                    // [fn, -input_size, first_task, tile_count, row_size,
                    //  activation (optional), weight rows...]
                    m_size = -static_cast<int>(inbuffer[1]);
                    matmul_tile_start_task = static_cast<int>(inbuffer[2]);
                    matmul_tile_count = static_cast<int>(inbuffer[3]);
                    matmul_tile_weight_size = static_cast<int>(inbuffer[4]);
                    const size_t activation_offset = 5;
                    const size_t weight_offset = activation_offset +
                        (matmul_include_activation ? static_cast<size_t>(m_size) : 0);
                    if (matmul_include_activation) {
                        infeature.assign(inbuffer.begin() + activation_offset,
                                         inbuffer.begin() + weight_offset);
                        matmul_activation_valid = true;
                        matmul_activation_row = matmul_tile_start_task / net->o_x;
                    } else {
                        assert(matmul_activation_valid &&
                               matmul_activation_row == matmul_tile_start_task / net->o_x &&
                               "Missing activation for weight tile");
                    }
                    weight.assign(inbuffer.begin() + weight_offset, inbuffer.end());
                } else {
                    infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 2 + m_size);
                    weight.assign(inbuffer.begin() + 2 + m_size, inbuffer.end()); //w + b
                }
            }
            else if (fn == 8 || fn == 12)           // max or avg pooling [fn] [map size] [i]
            {
                ch_size = 1;
                m_size = inbuffer[1];
                infeature.assign(inbuffer.begin() + 2, inbuffer.end());
                assert((infeature.size() == m_size) && "Inbuffer not correct after request (pooling)");
            }
            else if (fn >= MATMUL && fn <= GEGLU) // Layer Transformer
            {
                ch_size = 1;
                if (fn == MATMUL || fn == ADD || fn == SWIGLU || fn == GEGLU) { 
                    m_size = inbuffer[1];
                    if (fn == ADD || fn == SWIGLU || fn == GEGLU) {
                        // Interleaved operand pairs occupy activation SRAM; no weights.
                        infeature.assign(inbuffer.begin() + 2, inbuffer.end());
                        assert(infeature.size() == 2 * (use_elementwise_batch ? elementwise_tasks.size() : 1));
                    } else if (fn == MATMUL && use_matmul_tiling && inbuffer[1] < 0) {
                        // Tiled MatMul response uses the same layout as tiled FC.
                        m_size = -static_cast<int>(inbuffer[1]);
                        matmul_tile_start_task = static_cast<int>(inbuffer[2]);
                        matmul_tile_count = static_cast<int>(inbuffer[3]);
                        matmul_tile_weight_size = static_cast<int>(inbuffer[4]);
                        const size_t activation_offset = 5;
                        const size_t weight_offset = activation_offset +
                            (matmul_include_activation ? static_cast<size_t>(m_size) : 0);
                        if (matmul_include_activation) {
                            infeature.assign(inbuffer.begin() + activation_offset,
                                             inbuffer.begin() + weight_offset);
                            matmul_activation_valid = true;
                            matmul_activation_row = matmul_tile_start_task / net->o_x;
                        } else {
                            assert(matmul_activation_valid &&
                                   matmul_activation_row == matmul_tile_start_task / net->o_x &&
                                   "Missing activation for weight tile");
                        }
                        weight.assign(inbuffer.begin() + weight_offset, inbuffer.end());
                    } else {
                        // MatMul, LayerNorm, Add, RMSNorm
                        infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 2 + m_size);
                        weight.assign(inbuffer.begin() + 2 + m_size, inbuffer.end());
                    }
                } else if (fn == LAYERNORM || fn == RMSNORM) {
                    m_size = 1;
                    infeature.assign(inbuffer.begin() + 2, inbuffer.end()); 
                } else if (fn == SOFTMAX_TR) {                                      // Softmax
                    m_size = inbuffer[1];
                    causal_mask = inbuffer[2];
                    infeature.assign(inbuffer.begin() + 3, inbuffer.end());
                } else if (fn == EMBEDDING) {                                       // Embedding
                    m_size = inbuffer[1];
                    weight.assign(inbuffer.begin() + 2, inbuffer.end());
                } else if (fn == ROPE) {
                    infeature.assign(inbuffer.begin() + 4, inbuffer.begin() + 6);
                } else if (fn == ATTENTION) {                                       // Attention Hardware (Fused)
                    m_size = inbuffer[1] / inbuffer[3]; // query head dimension
                    infeature.assign(inbuffer.begin() + 9, inbuffer.begin() + 9 + m_size);
                }
            }

            recordSramUsage();
            if (use_matmul_tiling && (fn == MATMUL || (fn >= 4 && fn <= 7)) &&
                inbuffer[1] < 0) {
                assert(matmul_tile_count > 0 && "Invalid MatMul/Linear tile count");
                assert(matmul_tile_weight_size > 0 && "Invalid MatMul/Linear tile row size");
                assert(weight.size() == static_cast<size_t>(matmul_tile_count * matmul_tile_weight_size) &&
                       "Weight tile payload size mismatch");
                assert(infeature.size() == static_cast<size_t>(m_size) &&
                       "Activation tile payload size mismatch");
                assert(weight.size() <= MAC_WEIGHT_SRAM_LIMIT &&
                       "Weight tile exceeds MAC weight SRAM limit");
                assert(infeature.size() <= MAC_INPUT_SRAM_LIMIT &&
                       "Activation tile exceeds MAC input SRAM limit");
            }

            assert(infeature.size() <= MAC_INPUT_SRAM_LIMIT && "Input feature size exceeds MAC input SRAM limit");
            assert(weight.size() <= MAC_WEIGHT_SRAM_LIMIT && "Weight size exceeds MAC weight SRAM limit");

            // The receive buffer and activation vector represent the same data SRAM.
            if (use_elementwise_batch) inbuffer.clear();

            outfeature = 0.0;
            selfstatus = 3;
            pecycle = context.cycles;
            return;
        }
        else if(selfstatus == 3){
            if (compute_unit->busy()) return;
#ifdef cNoC_MODE
            // In cNoC mode, the MAC units act like co-processors for global reductions.
            // They should not execute MatMul, Add, or SwiGLU operations, as those are performed in-transit on the routers.
            if (fn == MATMUL || fn == ADD || fn == SWIGLU || fn == GEGLU) {
                cout << "FATAL ERROR: The MAC node " << id << " does not support executing function " 
                     << fn << " in cNoC_MODE! This operation is performed in-transit on the routers." << endl;
                outfeature = 0.0;
                selfstatus = 4;
                scheduleCompute(1);
                inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                return;
            }
#endif
            // normal MAC op
            if (fn >=0 && fn <=3) {                         // Conv
                float temp_sum = psum_accumulator; 
                
                temp_sum = ComputeKernels::dot(infeature.data(), weight.data(), ch_size, temp_sum);
                
                if (current_chunk == total_chunks - 1) {
                    temp_sum += weight.back();
                }

                outfeature = temp_sum;
                psum_accumulator = temp_sum;
            }
            else if (fn >= 4 && fn <= 7)                    // FC
            {
                if (use_matmul_tiling && matmul_tile_count > 0) {
                    pending_acks = matmul_tile_count;
                } else {
                    outfeature = ComputeKernels::dot(infeature.data(), weight.data(), m_size, outfeature);
                    outfeature += weight[m_size];
                    pending_acks = 1;
                }
            }
            else if (fn == 8)                               // max pooling
            {
                outfeature = ComputeKernels::pool(infeature.data(), m_size, false);
                selfstatus = 4;                             // ready for this computation
                scheduleCompute(1);

                inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                return;
            }
            else if (fn == 12)                              // average pooling
            {
                outfeature = ComputeKernels::pool(infeature.data(), m_size, true);
                selfstatus = 4;                             // ready for this computation
                scheduleCompute(1);

                inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                //packet_id++;
                return;
            }
            else if (fn >= MATMUL && fn <= GEGLU)                                               // Operazioni Transformer
            {
                if (use_elementwise_batch && (fn == ADD || fn == SWIGLU || fn == GEGLU)) {
                    pending_acks = static_cast<int>(elementwise_tasks.size());
                    received_acks = 0;
                    selfstatus = 4;
                    send = 1; // Earlier scalar results may return before the batch finishes.
                    // Reserve the whole batch; individual results retain their
                    // original offsets within it, including equal-ready ADDs.
                    scheduleCompute(LegacyComputeTiming::peElementwiseBatch(fn, elementwise_tasks.size() - 1, context.compute.pe));
                    for (size_t index = 0; index < elementwise_tasks.size(); ++index) {
                        const float gate = infeature[2 * index];
                        const float up = infeature[2 * index + 1];
                        outfeature = ComputeKernels::elementwisePair(fn, gate, up);
                        // Reuse consumed input space; output packets model timed egress.
                        infeature[index] = outfeature;
                        tmpm = elementwise_tasks[index];
                        pecycle = compute_unit->scheduled().started_at + LegacyComputeTiming::peElementwiseBatch(fn, index, context.compute.pe);
                        inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id],
                               context.packet_id_base + tmpm, id);
                    }
                    return;
                }
                if (fn == MATMUL) {                                                                 // MatMul
                    if (use_matmul_tiling && matmul_tile_count > 0) {
                        pending_acks = matmul_tile_count;
                    } else {
                        outfeature = ComputeKernels::dot(infeature.data(), weight.data(), m_size, outfeature);

                        #if USE_BIAS
                            outfeature += weight[m_size];
                        #endif
                        pending_acks = 1;
                    }
                } 
                else if (fn != ATTENTION) {
                    outfeature = computeTransformerElement();
                }
                else { // ATTENTION: cache updates and arithmetic precede timed result delivery.
                    const auto work = computeAttentionElement();
                    selfstatus = 4;
                    scheduleCompute(LegacyComputeTiming::peAttention(work, context.compute.pe));
                    inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);


                    return;
                }

                const Cycle calctime = LegacyComputeTiming::peTransformer(
                    fn, m_size, use_matmul_tiling && matmul_tile_count > 0 ? matmul_tile_count : 1, context.compute.pe);

                selfstatus = 4; // ready for output
                scheduleCompute(calctime);
                matmul_tile_compute_cycles = calctime;

                // Send one result per output task. The tile pays one compute
                // startup, but still produces one architectural output each.
                if (use_matmul_tiling && fn == MATMUL && matmul_tile_count > 0) {
                    const int row_size = matmul_tile_weight_size;
                    for (int tile_idx = 0; tile_idx < matmul_tile_count; tile_idx++) {
                        const int weight_offset = tile_idx * row_size;
                        outfeature = ComputeKernels::dot(infeature.data(), weight.data() + weight_offset, m_size);
                        #if USE_BIAS
                            if (row_size > m_size) outfeature += weight[weight_offset + m_size];
                        #endif
                        tmpm = matmul_tile_start_task + tile_idx;
                        inject(MessageKind::Result, dest_mem_id, 1, outfeature,
                               net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmpm, id);
                    }
                } else {
                    inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                }

                return;
            }

            int operation_count = ch_size * m_size;
            if (use_matmul_tiling && (fn >= 4 && fn <= 7) && matmul_tile_count > 0) {
                operation_count *= matmul_tile_count;
            }
            const Cycle calctime = LegacyComputeTiming::peConvOrDense(
                operation_count, fn % 4 == 0 && current_chunk == total_chunks - 1, context.compute.pe);
            scheduleCompute(calctime);

            if (current_chunk < total_chunks - 1) {
                selfstatus = 41;
                return;
            }

            // activation
            if ((fn % 4) == 0) //linear
            {
                selfstatus = 4; // ready for this computation
            }
            else if ((fn % 4) == 1)
            {
                // activation (relu)
                // cout << "from mac " << id << " output " << outfeature << endl;
                relu(outfeature);
                selfstatus = 4; // ready for output
            }
            else if ((fn % 4) == 2)
            {
                // activation (tanh)
                tanh(outfeature);
                selfstatus = 4; // ready for output
            }
            else if ((fn % 4) == 3)
            {
                // activation (sigmoid)
                sigmoid(outfeature);
                selfstatus = 4; // ready for output
            }
            else
            {
                outfeature = 0.0;
                selfstatus = 0; // back to initial state
                pecycle = context.cycles + 2; // sync cycles
                assert((0 < 1) && "Wrong function (fn)");
                return;
            }

            // inject
#ifndef newpooling
            if (use_matmul_tiling && (fn >= 4 && fn <= 7) && matmul_tile_count > 0) {
                const int row_size = matmul_tile_weight_size;
                for (int tile_idx = 0; tile_idx < matmul_tile_count; tile_idx++) {
                    const int weight_offset = tile_idx * row_size;
                    outfeature = ComputeKernels::dot(infeature.data(), weight.data() + weight_offset, m_size);
                    // Dense layers in the legacy model always carry a bias.
                    if (row_size > m_size) outfeature += weight[weight_offset + m_size];
                    tmpm = matmul_tile_start_task + tile_idx;
                    inject(MessageKind::Result, dest_mem_id, 1, outfeature,
                           net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmpm, id);
                }
            } else {
                inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
            }
            //packet_id++;
#endif
#ifdef newpooling
            if (this->npoolflag == 1){ //with pooling send out
                //only if pooling is needed
                if(this->n_tmpch >= 0){
                    for (int c : this->n_tmpm) // c is the dest id of it
                    {
                        this->tmpm = c;
                        this->tmpch = this->n_tmpch;
                        inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                        //packet_id++;
                    }
                }
            }
            else{ // normal send out
                inject(MessageKind::Result, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], context.packet_id_base + tmp_request, id);
                //packet_id++;
            }
#endif

            //added to reduce transmission delay
            //this->send = 2;
            return;
        }
        else if(selfstatus == 4){
#ifndef only3type
            if(((this->use_matmul_tiling || this->use_elementwise_batch) && this->pending_acks > 0 &&
                this->received_acks >= this->pending_acks) ||
               (!this->use_matmul_tiling && !this->use_elementwise_batch && this->send == 2)) // tile/result confirmation
            {
                this->send = 0;
                this->pending_acks = 0;
                this->received_acks = 0;
                if(this->routing_table.size()==0)
                {
                    this->selfstatus = 5;
                }
                else
                {
                    this->selfstatus = 0;               // back to initial state
                }
                //cout << "from mac " << this->id << " output " << this->outfeature << " " << selfstatus << endl;
                const int next_row = this->routing_table.empty() ? -1 :
                    this->routing_table.front() / net->o_x;
                const int current_row = this->matmul_tile_start_task / net->o_x;
                const bool preserve_activation =
                    this->use_matmul_tiling && this->matmul_activation_valid &&
                    next_row == current_row;
                this->weight.clear();
                if (!preserve_activation) {
                    this->infeature.clear();
                }
                this->inbuffer.clear();
                this->outfeature = 0.0;
                this->outfeature_vec.clear();
                if (this->use_matmul_tiling && this->matmul_activation_valid &&
                    next_row != current_row) {
                    this->matmul_activation_valid = false;
                    this->matmul_activation_row = -1;
                }
#ifdef newpooling
            this->npoolflag = 0;
            this->n_tmpch = -1;
            this->n_tmpm.clear();
#endif
                this->pecycle = context.cycles + 1; //cycles + 1
                return;
            }
            this->send = 1; // change from 1
            pecycle = context.cycles;
#endif
#ifdef only3type
            if (use_elementwise_batch && received_acks < pending_acks) return;
            this->send = 0;
            if(this->routing_table.size()==0)
            {
                this->selfstatus = 5;
                if (use_elementwise_batch) this->send = 3;
            }
            else
            {
                this->selfstatus = 0;               // back to initial state
            }
            this->current_chunk = 0;
            this->total_chunks = 1;
            this->psum_accumulator = 0.0f;

            //cout << "from mac " << this->id << " output " << this->outfeature << " " << selfstatus << endl;
            this->weight.clear();
            this->infeature.clear();
            this->inbuffer.clear();
            this->outfeature = 0.0;
#ifdef newpooling
            this->npoolflag = 0;
            this->n_tmpch = -1;
            this->n_tmpm.clear();
#endif
            this->pecycle = context.cycles + 1; //cycles + 1
            return;
#endif
        }
        else if (selfstatus == 41) {
            current_chunk++;
            selfstatus = 1;
            pecycle = context.cycles;
            weight.clear();
            infeature.clear();
            inbuffer.clear();
            return;
        }
    }
}

// Destructor
MAC::~MAC (){
    for (Packet* packet : pending_tx) Packet::release(packet);
}
