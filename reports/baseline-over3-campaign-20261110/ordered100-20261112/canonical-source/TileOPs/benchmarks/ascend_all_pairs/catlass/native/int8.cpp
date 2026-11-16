// Original example57 INT8 full-dequant template; public BF16 output selection.
#define CATLASS_JIT_ELEMENT_D bfloat16_t
#define CATLASS_JIT_X1_QUANT_MODE 4
#define CATLASS_JIT_X2_QUANT_MODE 2
#define KERNEL_NAME allpairs_catlass_int8_kernel
#define KERNEL_TYPE __mix__(1,2)
#include "57_ascend950_matmul_full_dequant/matmul_full_dequant_impl.cpp"
#include <tiling/platform/platform_ascendc.h>
extern "C" {
WorkspaceAllocFn g_catlassWorkspaceAlloc = nullptr;
WorkspaceFreeFn g_catlassWorkspaceFree = nullptr;
WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost = nullptr;
int allpairs_catlass_int8(void* a, void* b, void* sa, void* sb, void* out,
                          uint32_t m, uint32_t n, uint32_t k, void* stream) {
    if (!a || !b || !sa || !sb || !out || !m || !n || !k) return 1;
    CatlassKernel::MatmulParams params;
    params.m = m; params.n = n; params.k = k;
    params.inputAddr = {static_cast<uint8_t*>(a), static_cast<uint8_t*>(b),
                        static_cast<uint8_t*>(sa), static_cast<uint8_t*>(sb)};
    params.outputAddr = {static_cast<uint8_t*>(out)};
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    run(cores, stream, &params);
    return 0;
}
}
