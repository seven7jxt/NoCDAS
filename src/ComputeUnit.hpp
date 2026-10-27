#ifndef NOCDAS_COMPUTE_UNIT_HPP_
#define NOCDAS_COMPUTE_UNIT_HPP_

#include "SimulationTime.hpp"
#include <algorithm>

struct ComputeRequest {
    Cycle duration;                 // NoC cycles, supplied by the timing policy.
    Cycle operands_ready = 0;       // May include a pending MC read.
};

struct ComputeCompletion {
    Cycle started_at = 0;
    Cycle ready_at = 0;
};

// Resource scheduling only: operands, numerical kernels and packet ownership
// remain with the caller. Legacy PE/router kernels run at issue and attach the
// scheduled ready time to their results; terminal kernels run at completion.
class ComputeUnit {
public:
    virtual ~ComputeUnit() = default;
    virtual bool tryIssue(const ComputeRequest& request, Cycle now) = 0;
    virtual bool tryComplete(Cycle now, ComputeCompletion& completion) = 0;
    virtual bool busy() const = 0;
    virtual const ComputeCompletion& scheduled() const = 0;
};

// One reservation, including its completion slot. The owner must consume the
// completion before issuing again. No operand copies or per-operation allocation.
class SerialComputeUnit final : public ComputeUnit {
public:
    bool tryIssue(const ComputeRequest& request, Cycle now) override {
        if (pending_) return false;
        scheduled_.started_at = std::max(now, request.operands_ready);
        scheduled_.ready_at = scheduled_.started_at + request.duration;
        pending_ = true;
        return true;
    }

    bool tryComplete(Cycle now, ComputeCompletion& completion) override {
        if (!pending_ || now < scheduled_.ready_at) return false;
        completion = scheduled_;
        pending_ = false;
        return true;
    }

    bool busy() const override { return pending_; }
    // Retained after completion for timed egress and occupancy accounting.
    const ComputeCompletion& scheduled() const override { return scheduled_; }

private:
    ComputeCompletion scheduled_;
    bool pending_ = false;
};

#endif
