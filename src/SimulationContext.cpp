#include "SimulationContext.hpp"
#include "NoC/NetworkStorage.hpp"

SimulationContext::SimulationContext() : objects(std::make_unique<NetworkStorage>(*this)) {}
SimulationContext::~SimulationContext() = default;
