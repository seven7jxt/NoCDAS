#ifndef NOCDAS_SIMULATION_HPP_
#define NOCDAS_SIMULATION_HPP_

#include <memory>
#include <string>
#include <vector>
#include "parameters.hpp"
#include "SimulationContext.hpp"
#include "SimulationResult.hpp"
#include <iosfwd>

class Model;
class VCNetwork;
class MACnet;

struct SimulationOptions {
    std::string model_file = DEFAULT_NNMODEL_FILENAME;
    std::string weight_file = DEFAULT_NNWEIGHT_FILENAME;
    std::string input_file = DEFAULT_NNINPUT_FILENAME;
    size_t packet_trace_limit = CountNum;
    Cycle max_cycles = 50000000000ULL;
    ComputeResources compute;
};

// Owns one execution. The network step ordering and readiness rules are legacy.
class Simulation {
public:
    explicit Simulation(SimulationOptions options = {});
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    StopReason run();
    const SimulationContext& context() const { return context_; }
    StopReason status() const { return result_.status; }
    void reportLegacy() const;
    const SimulationResult& result() const { return result_; }
    void writeJson(std::ostream& out) const;
    void writeLayerCsv(std::ostream& out) const;
    void writePacketTraceCsv(std::ostream& out) const;
    const MACnet& accelerator() const;
    const VCNetwork& network() const;
private:
    SimulationContext context_;
    SimulationOptions options_;
    SimulationResult result_;
    std::vector<int> ni_counts_;
    std::unique_ptr<Model> cnnmodel;
    std::unique_ptr<VCNetwork> vcNetwork;
    std::unique_ptr<MACnet> macnet;
};

#endif
