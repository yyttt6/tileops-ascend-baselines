// Expose the original example's existing FP16/BF16 and optional-bias choices.
#define main classic_opstensor_unused_cli_main
#include "mat_mul/mat_mul_basic/mat_mul_basic.cpp"
#undef main

extern "C" int classic_opstensor_gemm_cases(
    void* a, void* b, void* out, void* bias,
    int64_t m, int64_t n, int64_t k, int dtype, void* stream)
{
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0 || (dtype != 0 && dtype != 1)) return 1;
    TilingConfig config = GetTilingConfig(dtype == 0 ? "float16" : "bfloat16");
    BasicTiling tiling = ComputeTiling(config, m, k, n);
    int64_t cores = GetAicCoreNum();
    if (cores <= 0) return 2;
    LaunchParams p = {
        static_cast<uint8_t*>(a), static_cast<uint8_t*>(b), static_cast<uint8_t*>(out),
        static_cast<uint8_t*>(bias), m, n, k, cores, &tiling, &config, stream, false, false, false,
    };
    if (dtype == 0)
        LaunchKernel<half, half, half, half, false, false, false>(p);
    else
        LaunchKernel<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, false>(p);
    return 0;
}
