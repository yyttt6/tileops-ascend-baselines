// Host ABI for the unchanged CATLASS example-49 kernel and tiling.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "../../../../ascend_baselines/catlass/examples/49_ascend950_flash_attention_infer/fai_kernel.h"
#include "../../../../ascend_baselines/catlass/examples/49_ascend950_flash_attention_infer/fai_tiling.h"

extern "C" uint64_t classic_catlass_fa_cases_tiling_size() {
    return sizeof(FATilingData);
}

extern "C" int classic_catlass_fa_cases_prepare(void* host_tiling,
                                        const int64_t* sizes, float scale) {
    FAInferTiling::FAInfo info{};
    info.batchSize = sizes[0];
    info.seqSize = sizes[1];
    info.seqInnerSize = sizes[2];
    info.numOfHeads = sizes[3];
    info.numOfKVHeads = sizes[4];
    info.headSize = sizes[5];
    info.maskType = sizes[6];
    info.blockSize = 128;
    info.maxBlockNumPerBatch = (sizes[2] + 127) / 128;
    info.scaleValue = scale;
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto* tiling = static_cast<FATilingData*>(host_tiling);
    *tiling = FATilingData{};
    return FAInferTiling::GetFATilingParam(info, cores, *tiling);
}

extern "C" int classic_catlass_fa_cases_run(void* q, void* k, void* v, void* out,
    void* q_lengths, void* kv_lengths, void* tiling, void* mask, void* stream, int dtype) {
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (dtype == 0 && mask == nullptr) {
        FAInferTla<half, false, false><<<cores, nullptr, stream>>>(
            static_cast<uint8_t*>(q), static_cast<uint8_t*>(k), static_cast<uint8_t*>(v),
            static_cast<uint8_t*>(mask), nullptr, static_cast<uint8_t*>(out),
            static_cast<uint8_t*>(q_lengths), static_cast<uint8_t*>(kv_lengths),
            static_cast<uint8_t*>(tiling));
        return 0;
    }
    if (dtype == 0 && mask != nullptr) {
        FAInferTla<half, true, false><<<cores, nullptr, stream>>>(
            static_cast<uint8_t*>(q), static_cast<uint8_t*>(k), static_cast<uint8_t*>(v),
            static_cast<uint8_t*>(mask), nullptr, static_cast<uint8_t*>(out),
            static_cast<uint8_t*>(q_lengths), static_cast<uint8_t*>(kv_lengths),
            static_cast<uint8_t*>(tiling));
        return 0;
    }
    if (dtype == 1 && mask == nullptr) {
        FAInferTla<bfloat16_t, false, false><<<cores, nullptr, stream>>>(
            static_cast<uint8_t*>(q), static_cast<uint8_t*>(k), static_cast<uint8_t*>(v),
            static_cast<uint8_t*>(mask), nullptr, static_cast<uint8_t*>(out),
            static_cast<uint8_t*>(q_lengths), static_cast<uint8_t*>(kv_lengths),
            static_cast<uint8_t*>(tiling));
        return 0;
    }
    if (dtype == 1 && mask != nullptr) {
        FAInferTla<bfloat16_t, true, false><<<cores, nullptr, stream>>>(
            static_cast<uint8_t*>(q), static_cast<uint8_t*>(k), static_cast<uint8_t*>(v),
            static_cast<uint8_t*>(mask), nullptr, static_cast<uint8_t*>(out),
            static_cast<uint8_t*>(q_lengths), static_cast<uint8_t*>(kv_lengths),
            static_cast<uint8_t*>(tiling));
        return 0;
    }
    return -1;
}
