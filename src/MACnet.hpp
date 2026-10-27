/*
 * MACnet.hpp
 * Global control and MC functions
 */

#ifndef MACNET_HPP_
#define MACNET_HPP_

#include "SimulationContext.hpp"
#include "ComputeUnit.hpp"

#include <cmath>
#include <map>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <algorithm>    //std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include "MAC.hpp"
#include "Model.hpp"
#include "NoC/VCNetwork.hpp"

using namespace std;





// NoC
class VCNetwork;

class MAC;

class MACnet
{
public:
  SimulationContext& context;

  /** @brief MAC Network
   *
   */
	MACnet (int mac_num, int t_pe_x, int t_pe_y, Model *m, VCNetwork* t_Network);
	std::vector<MAC*> MAC_list;
	VCNetwork* vcNetwork;

	void create_input();
	int weight_row_offset(int layer_id) const;
	vector<vector<float>> weight_table;
	vector<vector<float>> input_table;
	vector<vector<float>> output_table;
	std::map<int, vector<vector<float>>> layer_outputs_history;
	vector<float> precalc_mean;
	vector<float> precalc_var;
	vector<float> precalc_rms;

	deque< deque< int > > mapping_table;
	void mapping(int neuronnum);
	void matmul_tile_mapping(int neuronnum);
	void ymapping(int neuronnum);
	void rmapping(int neuronnum);

	void runOneStep();
	void checkStatus();
    void reportSramPeaks(std::ostream& out) const;

	Model* cnnmodel;
	int macNum;
	int pe_x;
	int pe_y;
	int used_pe;

	int c_layer; 		// current layer
	int n_layer; 		// total layer

	int in_ch; 			// for input channel
	int in_x;
	int in_y;

	// for new pooling
	int no_x;
	int no_y;
	int nw_x;
	int nw_y;
	int no_ch; 			// next o_ch in pooling layer
	int npad; 			// padding
	int nstride; 		// stride
	// vector<vector<float>> pooling_table;

	int w_ch; 			// for filter
	int w_x; 
	int w_y;
	int st_w;
	int pad;
	int stride;

	int o_ch; 			// for output
	int o_x; 
	int o_y;

	int o_fn; 			// for function

	int readyflag;
    int causal_mask_flag;

	// for print
	vector<Cycle> Layer_latency;
    
    // 0 = Idle, 1 = Distributing, 2 = Injecting compute, 3 = Waiting for results.
    // Results may return during phase 2; count them independently of injection.
    int cnoc_phase;
    size_t cnoc_send_index = 0;
    int cnoc_results_remaining = 0;
    Packet* cnoc_pending_tx = nullptr;
    
	// Sorted path of routers that will perform in-transit computation
    std::deque<int> cnoc_compute_path;

	int cnoc_current_chunk = 0;
    int cnoc_total_chunks = 1;
    int cnoc_chunk_size = 0;
    bool cnoc_output_tiling = false;
    int cnoc_output_batch_size = 0;
    int cnoc_output_offset = 0;
    int cnoc_output_count = 0;
    void map_cnoc_output_chunk();
    
	// Mapping of weights to router for preparing the distribution phase
    void cNoC_mapping(int task_num);
    
	// It generates the distribution packets (type 4) and computation packets (type 5)
    void inject_cNoC_traffic(); 

    struct TerminalLane {
        std::deque<int> tasks;
        // Abstract MC receive queue, partitioned by execution unit; not SRAM.
        std::deque<float> inbuffer;
        size_t issued = 0;
        size_t batch_count = 0;
        SerialComputeUnit compute_unit;
        bool waiting_for_input = false;
        void issue(Cycle now, Cycle operands_ready);
    };
    std::vector<std::vector<TerminalLane>> terminal_lanes;
    std::vector<std::uint64_t> terminal_outputs;
    int terminal_remaining = 0;
    bool terminal_gate_active = false;
    Cycle terminal_dispatch_ready = 0;
    void start_terminal_gate();
    void run_terminal_gate();

	~MACnet ();
private:
    bool reported_nonfinite = false;
    std::vector<bool> is_dest = std::vector<bool>(TOT_NUM, false);
    bool init_dest = false;
};

#endif /* MACNET_HPP_ */
