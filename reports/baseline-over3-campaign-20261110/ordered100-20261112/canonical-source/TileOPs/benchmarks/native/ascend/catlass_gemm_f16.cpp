// Instantiate CATLASS's unchanged example-43 template and expose a plain host ABI.
#define CATLASS_JIT_ELEMENT_A half
#define CATLASS_JIT_ELEMENT_B half
#define CATLASS_JIT_ELEMENT_C half
// Original host chooses 31 for M <= N (the fixed square pilot).
#define CATLASS_JIT_BLOCK_SCHEDULER 31
#define KERNEL_NAME classic_catlass_gemm_f16_kernel
#define KERNEL_TYPE __cube__
#include "43_ascend950_basic_matmul/ascend950_basic_matmul_impl.cpp"
#include <tiling/platform/platform_ascendc.h>

extern "C" {
WorkspaceAllocFn g_catlassWorkspaceAlloc = nullptr;
WorkspaceFreeFn g_catlassWorkspaceFree = nullptr;
WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost = nullptr;

int classic_catlass_gemm_f16(
    void* a, void* b, void* out, uint32_t m, uint32_t n, uint32_t k, void* stream)
{
    if (!a || !b || !out || m == 0 || n == 0 || k == 0 || m > n) return 1;
    CatlassKernel::MatmulParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.inputAddr = {static_cast<uint8_t*>(a), static_cast<uint8_t*>(b)};
    params.outputAddr = {static_cast<uint8_t*>(out)};
    const auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // BasicMatmulTla::GetWorkspaceSize is zero; no allocator is required.
    run(cores, stream, &params);
    return 0;
}
}
