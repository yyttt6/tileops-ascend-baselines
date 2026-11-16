// Host ABI for the original CATLASS example51 optest type-parameter API.
// Only public operand/accumulator/output types are selected; native schedule is unchanged.
#define CATLASS_JIT_ELEMENT_A float8_e4m3_t
#define CATLASS_JIT_ELEMENT_B float8_e4m3_t
#define CATLASS_JIT_ELEMENT_C float
#define CATLASS_JIT_ELEMENT_Y half
#define KERNEL_NAME allpairs_catlass_fp8_f16_kernel
#define KERNEL_TYPE __mix__(1,2)
#include "51_ascend950_quant_matmul_per_group_per_block_tla/quant_matmul_per_group_per_block_impl.cpp"
#include <tiling/platform/platform_ascendc.h>
extern "C" {
WorkspaceAllocFn g_catlassWorkspaceAlloc = nullptr;
WorkspaceFreeFn g_catlassWorkspaceFree = nullptr;
WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost = nullptr;
int allpairs_catlass_fp8_f16(void* a, void* b, void* sa, void* sb, void* bias, void* out,
                              uint32_t m, uint32_t n, uint32_t k, void* stream) {
    if (!a || !b || !sa || !sb || !out || !m || !n || !k) return 1;
    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
    // Existing public Arguments fields select the original optional bias epilogue.
    typename MatmulKernel::Arguments args{
        GemmCoord{m,n,k}, static_cast<uint8_t*>(a), layoutA,
        static_cast<uint8_t*>(b), layoutB, static_cast<uint8_t*>(out), layoutC,
        static_cast<uint8_t*>(bias),
        {static_cast<uint8_t*>(out), static_cast<uint8_t*>(sb), static_cast<uint8_t*>(sa),
         static_cast<uint8_t*>(bias), tla::get<0>(L0TileShape{}), tla::get<1>(L0TileShape{}),
         tla::get<2>(L0TileShape{}), 1, 128, 128, bias == nullptr ? 0u : 1u}};
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto tasks = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    Catlass::RunKernel<MatmulKernel>(args, stream, std::min(cores, tasks));
    return 0;
}
}
