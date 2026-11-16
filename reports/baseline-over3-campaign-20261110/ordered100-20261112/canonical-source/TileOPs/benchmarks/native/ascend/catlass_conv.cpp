// Thin ABI around the unchanged CATLASS example-56 host/kernel implementation.
#include "56_ascend950_basic_conv2d_tla/basic_conv2d_tla.cpp"
#include <tiling/platform/platform_ascendc.h>

extern "C" {
WorkspaceAllocFn g_catlassWorkspaceAlloc = nullptr;
WorkspaceFreeFn g_catlassWorkspaceFree = nullptr;
WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost = nullptr;

int classic_catlass_conv_f16(void* x, void* w, void* out, const uint32_t* sizes, void* stream)
{
    if (!x || !w || !out || !sizes) return 1;
    try {
        CatlassKernel::ConvParams p;
        // N, H, W, Cin, Cout; KH, KW; SH, SW; PL, PR, PT, PB; DH, DW
        p.fmapRelated.assign(sizes, sizes + 5);
        p.filterRelated.assign(sizes + 5, sizes + 7);
        p.strideList.assign(sizes + 7, sizes + 9);
        p.padList.assign(sizes + 9, sizes + 13);
        p.dilationList.assign(sizes + 13, sizes + 15);
        p.inputAddr = {static_cast<uint8_t*>(x), static_cast<uint8_t*>(w)};
        p.outputAddr = {static_cast<uint8_t*>(out)};
        const auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        // BasicConv2dTla::GetWorkspaceSize is zero.
        CatlassKernel::Ascend950BasicConv2dTLA(cores, stream, p);
        return 0;
    } catch (const std::exception&) {
        return 2;
    }
}
}
