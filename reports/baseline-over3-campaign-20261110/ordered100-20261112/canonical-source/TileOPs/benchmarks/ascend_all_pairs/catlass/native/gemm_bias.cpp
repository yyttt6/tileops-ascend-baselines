// Host binding for the original example43 BasicMatmulTla bias template.
// All native kernel, copy, accumulation and schedule code remains in CATLASS.
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include <tiling/platform/platform_ascendc.h>

using namespace Catlass;

template<class Element, int Direction>
int LaunchBias(void* a, void* b, void* bias, void* out,
               uint32_t m, uint32_t n, uint32_t k, aclrtStream stream)
{
    using ArchTag = Arch::Ascend950;
    // The existing example43 defaults, including both original swizzle branches.
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<32>>;
    using Layout = layout::RowMajor;
    // Exact widening of the public bias to the native accumulator type.
    using ElementBias = float;
    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, Element, Layout, Element, Layout, Element, Layout, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, Element, Element, Element, ElementBias, TileCopy>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, Direction>;
    using Kernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, void, Scheduler>;
    typename Kernel::Arguments args{
        GemmCoord{m, n, k}, static_cast<uint8_t*>(a), tla::MakeLayout<Element, Layout>(m, k),
        static_cast<uint8_t*>(b), tla::MakeLayout<Element, Layout>(k, n),
        static_cast<uint8_t*>(out), tla::MakeLayout<Element, Layout>(m, n), static_cast<uint8_t*>(bias)};
    using Adapter = Gemm::Device::DeviceGemm<Kernel>;
    if (Adapter::CanImplement(args) != Status::kSuccess || Adapter::GetWorkspaceSize(args) != 0) return 2;
    Adapter op;
    op.Initialize(args, nullptr);
    const uint32_t cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t tasks = CeilDiv(m, uint32_t(256)) * CeilDiv(n, uint32_t(256));
    op(stream, cores < tasks ? cores : tasks);
    return 0;
}

extern "C" int classic_catlass_gemm_bias(void* a, void* b, void* bias, void* out,
                                        uint32_t m, uint32_t n, uint32_t k, int dtype, void* stream)
{
    if (!a || !b || !bias || !out || !m || !n || !k) return 1;
    if (dtype == 0) {
        return m > n ? LaunchBias<half, 0>(a, b, bias, out, m, n, k, stream)
                     : LaunchBias<half, 1>(a, b, bias, out, m, n, k, stream);
    }
    if (dtype == 1) {
        return m > n ? LaunchBias<bfloat16_t, 0>(a, b, bias, out, m, n, k, stream)
                     : LaunchBias<bfloat16_t, 1>(a, b, bias, out, m, n, k, stream);
    }
    return 1;
}
