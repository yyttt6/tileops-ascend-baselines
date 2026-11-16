// Thin ABI for the unchanged ATVC ReduceSumCustom and original policy dispatch.
#include <cstdint>
#include <vector>
#include <new>
#include "reduce_sum_impl.h"
struct Plan { ATVC::ReduceParam param{}; ATVC::ReducePolicy policy{-1,-1,-1}; };
extern "C" void* atvc_sum_plan(const int64_t* sizes, int rank, const int64_t* axes, int count,
                                uint64_t* workspace_bytes) {
    auto* plan = new(std::nothrow) Plan;
    if (!plan) return nullptr;
    if (!ATVC::Host::CalcReduceTiling<ReduceOpTraitsFloat>(
            std::vector<int64_t>(sizes, sizes + rank), std::vector<int64_t>(axes, axes + count),
            &plan->policy, &plan->param)) { delete plan; return nullptr; }
    *workspace_bytes = plan->param.workspaceSize;
    return plan;
}
extern "C" int atvc_sum_run(void* opaque, void* x, void* y, void* workspace, void* stream) {
    auto* plan = static_cast<Plan*>(opaque);
    if (!plan || !x || !y) return 1;
    auto param = plan->param;
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspace);
    if (plan->policy == ATVC::REDUCE_POLICY0) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY1) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY2) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY2><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY3) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY3><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY4) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY4><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY5) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY5><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY6) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY6><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY7) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY7><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY8) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY8><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY9) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY9><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY10) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY10><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY11) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY11><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY12) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY12><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY13) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY13><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY14) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY14><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY15) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY15><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY16) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY16><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY17) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY17><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY18) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY18><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY19) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY19><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY20) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY20><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY21) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY21><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else if (plan->policy == ATVC::REDUCE_POLICY22) {
        ReduceSumCustom<ReduceOpTraitsFloat, ATVC::REDUCE_POLICY22><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), param);
    }
    else return 2;
    return 0;
}
extern "C" void atvc_sum_destroy(void* opaque) { delete static_cast<Plan*>(opaque); }
