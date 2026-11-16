// Reuse the original example's template and host tiling selection verbatim.
// Its file-based CLI is compiled under another name and is never called.
#define main classic_opstensor_unused_cli_main
#include "mat_mul/mat_mul_basic/mat_mul_basic.cpp"
#undef main

extern "C" int classic_opstensor_gemm_bf16(
    void* a, void* b, void* out, int64_t m, int64_t n, int64_t k, void* stream)
{
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0) return 1;
    TilingConfig config = GetTilingConfig("bfloat16");
    BasicTiling tiling = ComputeTiling(config, m, k, n);
    int64_t cores = GetAicCoreNum();
    if (cores <= 0) return 2;
    LaunchParams p = {
        static_cast<uint8_t*>(a), static_cast<uint8_t*>(b), static_cast<uint8_t*>(out),
        nullptr, m, n, k, cores, &tiling, &config, stream, false, false, false,
    };
    LaunchKernel<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, false>(p);
    return 0;
}
