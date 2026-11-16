// Preserve the original AtlasA2 MLA host, tiling, kernels and schedule verbatim.
// This bridge probes whether the deployed3510 compiler can build that source.
#include <cstdint>
// Preserve the MLA constants before unrelated umbrella headers reuse the
// vendor's generic KERNEL_COMMON include guard. No constant value is changed.
#include "19_mla/tiling/kernel_common.hpp"
#include "19_mla/mla.cpp"
#include "19_mla/tiling/mla_tiling.cpp"
#include <tiling/platform/platform_ascendc.h>
extern "C" int allpairs_catlass_mla(
    void* q, void* qr, void* k, void* kr, void* pages, void* out,
    uint32_t batch, uint32_t heads, uint32_t length, int dtype, void* stream)
{
    if (!q || !qr || !k || !kr || !pages || !out || !batch || !length || length % 128 ||
        (heads != 64 && heads != 128) || (dtype != 0 && dtype != 1)) return 1;
    CatlassKernel::MlaParams params;
    params.batch = batch;
    params.qNtokens = batch;
    params.qSeqlen = 1;
    params.kvSeqlen = length;
    params.numHeads = heads;
    params.kvHeads = 1;
    params.embeddingSize = 512;
    params.qRopeHeadDim = 64;
    params.kvRopeHeadDim = 64;
    params.numBlocks = batch * (length / 128);
    params.blockSize = 128;
    params.dataType = dtype == 0 ? ACL_FLOAT16 : ACL_BF16;
    params.qSeqHost.assign(batch, 1);
    params.kvSeqHost.assign(batch, length);
    params.inputAddr = {static_cast<uint8_t*>(q), static_cast<uint8_t*>(qr), static_cast<uint8_t*>(k),
                        static_cast<uint8_t*>(kr), static_cast<uint8_t*>(pages)};
    params.outputAddr = {static_cast<uint8_t*>(out)};
    const auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    CatlassKernel::Mla(cores, stream, params);
    // The original host synchronizes and frees its own transient workspaces.
    return 0;
}
