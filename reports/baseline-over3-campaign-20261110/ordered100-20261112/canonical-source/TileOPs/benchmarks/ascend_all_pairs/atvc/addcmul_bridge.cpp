// Thin C ABI for ATVC's unchanged Addcmul example and both original policies.
#include <cstdint>
#include <new>
#include <vector>
#include "broadcast/host/broadcast_host.h"
#include "addcmul.h"

using Traits = ATVC::OpTraits<ATVC::OpInputs<float, float, float>,
                             ATVC::OpOutputs<float>, ATVC::OpTemps<float, float, float>>;
struct Plan {
    ATVC::BroadcastParam param{};
    ATVC::BroadcastPolicy policy{-1, -1, -1};
};

extern "C" void* atvc_addcmul_plan(int64_t count, uint64_t* workspace_bytes) {
    if (count <= 0 || !workspace_bytes) return nullptr;
    auto* plan = new(std::nothrow) Plan;
    if (!plan) return nullptr;
    plan->param.nBufferNum = 1; // Unchanged example buffer setting.
    // The original tiler requires a broadcast axis. Repeat one row and crop it
    // in the caller; this changes only layout, not the original computation.
    if (!ATVC::Host::CalcBroadcastTiling<Traits>(
            {1, count}, {2, count}, &plan->policy, &plan->param)) {
        delete plan;
        return nullptr;
    }
    *workspace_bytes = plan->param.workspaceSize;
    return plan;
}

extern "C" int atvc_addcmul_run(void* opaque, void* tensor1, void* tensor2,
                                void* input, void* output, float value,
                                void* workspace, void* stream) {
    auto* plan = static_cast<Plan*>(opaque);
    if (!plan || !tensor1 || !tensor2 || !input || !output) return 1;
    auto param = plan->param;
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspace);
    if (plan->policy == ATVC::BROADCAST_POLICY0) {
        AddcmulCustom<Traits, ATVC::BROADCAST_POLICY0><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(tensor1), static_cast<uint8_t*>(tensor2),
            static_cast<uint8_t*>(input), static_cast<uint8_t*>(output), param, value);
    } else if (plan->policy == ATVC::BROADCAST_POLICY1) {
        AddcmulCustom<Traits, ATVC::BROADCAST_POLICY1><<<param.tilingData.coreNum, nullptr, stream>>>(
            static_cast<uint8_t*>(tensor1), static_cast<uint8_t*>(tensor2),
            static_cast<uint8_t*>(input), static_cast<uint8_t*>(output), param, value);
    } else return 2;
    return 0;
}

extern "C" void atvc_addcmul_destroy(void* opaque) { delete static_cast<Plan*>(opaque); }
