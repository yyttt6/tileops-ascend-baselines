// Host-only entry point for ATVC's existing, unchanged AddCustom template.
// Kernel math, pipeline and tiling defaults are all supplied by the original header.
#include <cstdint>
#include <climits>
#include "add_custom_impl.h"

template <typename Scalar>
static int classic_atvc_add(
    void* x, void* y, void* output, int64_t elements, void* stream)
{
    using Traits = ATVC::OpTraits<ATVC::OpInputs<Scalar, Scalar>, ATVC::OpOutputs<Scalar>>;
    if (!x || !y || !output || elements <= 0 || elements > INT_MAX) {
        return 1;
    }
    ATVC::EleWiseParam param;
    if (!ATVC::Host::CalcEleWiseTiling<Traits>(static_cast<int32_t>(elements), param)) {
        return 2;
    }
    AddCustom<Traits><<<param.tilingData.blockNum, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(y),
        static_cast<uint8_t*>(output), param);
    return 0;
}

extern "C" int classic_atvc_add_bf16(void* x, void* y, void* out, int64_t n, void* stream)
{
    return classic_atvc_add<bfloat16_t>(x, y, out, n, stream);
}

extern "C" int classic_atvc_add_f16(void* x, void* y, void* out, int64_t n, void* stream)
{
    return classic_atvc_add<half>(x, y, out, n, stream);
}

extern "C" int classic_atvc_add_f32(void* x, void* y, void* out, int64_t n, void* stream)
{
    return classic_atvc_add<float>(x, y, out, n, stream);
}
