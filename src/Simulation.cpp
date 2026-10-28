#include "Simulation.hpp"
#include "MACnet.hpp"
#include "Model.hpp"
#include "NoC/VCNetwork.hpp"
#include <stdexcept>
#include <utility>

Simulation::Simulation(SimulationOptions options) : options_(std::move(options)) {}
Simulation::~Simulation() = default;

StopReason Simulation::run() {
    if (result_.status != StopReason::NotStarted || cnnmodel)
        throw std::logic_error("A Simulation instance can only run once");
    context_.cycles = 0;
    context_.compute = options_.compute;
    context_.packet_id_base = 0;
    context_.packet_trace_limit = options_.packet_trace_limit;
    cnnmodel = std::make_unique<Model>(options_.model_file, options_.weight_file, options_.input_file);
    cnnmodel->load();
#ifdef fulleval
    cnnmodel->loadin();
    cnnmodel->loadweight();
#elif defined randomeval
    cnnmodel->randomin();
    cnnmodel->randomweight();
#endif
    ni_counts_.assign(TOT_NUM, 1);
    const int flit_depth = INPORT_FLIT_BUFFER_SIZE;
    vcNetwork = std::make_unique<VCNetwork>(context_, TOT_NUM, X_NUM, TOT_NUM, ni_counts_.data(),
        VN_NUM, VC_PER_VN, VC_PRIORITY_PER_VN, flit_depth);
    macnet = std::make_unique<MACnet>(PE_X_NUM * PE_Y_NUM, PE_X_NUM, PE_Y_NUM,
        cnnmodel.get(), vcNetwork.get());
    Cycle layer_start = 0;
    for (; context_.cycles < options_.max_cycles; ++context_.cycles) {
        const int previous_layer = macnet->c_layer;
        macnet->checkStatus();
        if (macnet->c_layer != previous_layer) {
            result_.layers.push_back({previous_layer, macnet->c_layer - 1,
                                      layer_start, context_.cycles, true});
            layer_start = context_.cycles;
        }
        if (macnet->c_layer == macnet->n_layer) {
            result_.cycles = context_.cycles;
            result_.status = StopReason::Completed;
            return result_.status;
        }
        macnet->runOneStep();
        vcNetwork->runOneStep();
        if (context_.cycles % PRINT == 0) std::cout << "Running cycles: " << context_.cycles << std::endl;
    }
    result_.cycles = context_.cycles;
    result_.layers.push_back({macnet->c_layer, macnet->c_layer,
                              layer_start, context_.cycles, false});
    result_.status = StopReason::CycleLimit;
    return result_.status;
}

const MACnet& Simulation::accelerator() const {
    if (!macnet) throw std::logic_error("Simulation has not started");
    return *macnet;
}
const VCNetwork& Simulation::network() const {
    if (!vcNetwork) throw std::logic_error("Simulation has not started");
    return *vcNetwork;
}

void Simulation::reportLegacy() const {
    if (!macnet) throw std::logic_error("Simulation has not started");
#ifdef fulleval
    if (result_.status == StopReason::Completed) {
        for (float j: macnet->output_table[0]) {
            cout << j << ' ';
        }
        cout << endl;

        std::string filename = "output_baseline.txt";

#ifndef cNoC_MODE
        // Baseline execution
        std::ofstream out_file(filename);
        if (out_file.is_open()) {
            for (float j : macnet->output_table[0]) {
                out_file << j << " ";
            }
            out_file.close();
            cout << "\n=> [BASELINE] Output saved in " << filename << endl;
        }
#else
        // if we're using cNoC_MODE, we want to compare the output with the baseline
        std::ifstream in_file(filename);
        if (in_file.is_open()) {
            std::vector<float> baseline_out;
            float val;
            while (in_file >> val) {
                baseline_out.push_back(val);
            }
            in_file.close();

            if (baseline_out.size() == macnet->output_table[0].size()) {
                float dot_product = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
                float mae = 0.0f;
                int n = baseline_out.size();

                for (int i = 0; i < n; ++i) {
                    float a = baseline_out[i];
                    float b = macnet->output_table[0][i];
                    dot_product += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                    mae += std::abs(a - b);
                }

                float cosine_similarity = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
                mae /= n;

                cout << "\n=======================================================\n";
                cout << "   Precision Analysis (cNoC In-Transit vs Baseline)\n";
                cout << "=======================================================\n";
                cout << "Cosine Similarity : " << cosine_similarity * 100.0f << " %\n";
                cout << "Mean Absolute Err : " << mae << "\n";
                cout << "=======================================================\n";
            
                if (cosine_similarity > 0.98f) {
                    cout << "SUCCESS: Tensors are directionally equivalent!\n" << endl;
                } else {
                    cout << "WARNING: Structure mismatch detected.\n" << endl;
                }
            } else {
                cout << "\nError: Output dimensions do not match (" 
                     << baseline_out.size() << " vs " << macnet->output_table[0].size() << ")" << endl;
            }
        } else {
            cout << "\nWARNING: Run the simulation WITHOUT cNoC_MODE first to create the baseline!" << endl;
        }
#endif
    }
#endif

    cout << "Cycles: " << context_.cycles << endl;
    cout << "Packet id: " << context_.packet_id_base << endl;
    cout << "Flit size (bytes): " << FLIT_LENGTH << endl;
    cout << "Total inter-router flit-hops: " << vcNetwork->getTotalFlitHops() << endl;
    cout << "Total inter-router byte-hops: " << vcNetwork->getTotalByteHops() << endl;
    const WaitCounters wait_cycles = vcNetwork->getTotalWaitCycles();
    cout << "Total wait-cycles: pe-net=" << wait_cycles.pe_network
         << " pe-mem=" << wait_cycles.pe_memory
         << " router-mfu=" << wait_cycles.router_mfu
         << " router-kv=" << wait_cycles.router_kv << endl;
    cout << "Total KV evictions: " << vcNetwork->getTotalKVEvictions() << endl;
    macnet->reportSramPeaks(cout);

    if (result_.status == StopReason::CycleLimit)
        std::cout << "Simulation stopped: cycle_limit (incomplete)" << std::endl;
}

void Simulation::writePacketTraceCsv(std::ostream& out) const {
    writePacketTraces(out, context_.packet_traces);
}
