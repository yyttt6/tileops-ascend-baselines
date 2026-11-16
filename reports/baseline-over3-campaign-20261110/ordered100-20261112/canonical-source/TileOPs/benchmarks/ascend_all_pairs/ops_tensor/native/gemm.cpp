// The vendor kernel, host tiling and launch policy are included unmodified.
#define main allpairs_unused_gemm_main
#include "examples/mat_mul/mat_mul_basic/mat_mul_basic.cpp"
#undef main

extern "C" int allpairs_opstensor_gemm(
    void* a, void* b, void* out, void* bias,
    int64_t m, int64_t n, int64_t k, int dtype, void* stream)
{
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0 || dtype < 0 || dtype > 2) return 1;
    const char* type = dtype == 0 ? "float16" : (dtype == 1 ? "bfloat16" : "float32");
    TilingConfig config = GetTilingConfig(type);
    BasicTiling tiling = ComputeTiling(config, m, k, n);
    int64_t cores = GetAicCoreNum();
    if (cores <= 0) return 2;
    LaunchParams p = {
        static_cast<uint8_t*>(a), static_cast<uint8_t*>(b), static_cast<uint8_t*>(out),
        static_cast<uint8_t*>(bias), m, n, k, cores, &tiling, &config, stream, false, false, false,
    };
    if (dtype == 0)
        LaunchKernel<half, half, half, half, false, false, false>(p);
    else if (dtype == 1)
        LaunchKernel<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, false>(p);
    else
        LaunchKernel<float, float, float, float, false, false, false>(p);
    return 0;
}
