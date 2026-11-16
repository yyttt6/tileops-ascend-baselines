// Host ABI only: original example70 kernels and original tiling implementation.
#include <algorithm>
#include <new>
#define main classic_catlass70_unused_demo_main
#include "../../../../ascend_baselines/catlass/examples/70_ascend950_flash_attention_chunk_prefill/fai.cpp"
#undef main

extern "C" uint64_t classic_catlass70_tiling_size() {
    return sizeof(FAInferTilingData);
}

extern "C" uint64_t classic_catlass70_prepare(
    void* host, const int64_t* sizes, float scale, int dtype) {
    const int64_t batch = sizes[0], sq = sizes[1], sk = sizes[2];
    std::vector<int64_t> qLengths(batch), kvLengths(batch, sk);
    for (int64_t i = 0; i < batch; ++i) qLengths[i] = (i + 1) * sq;
    optiling::FAInferContext info{};
    info.numTokens = batch * sq;
    info.numHeads = sizes[3];
    info.kvHeads = sizes[4];
    info.embeddingSize = sizes[5];
    info.embeddingSizeV = 128;
    info.blockSize = 128;
    info.maxNumBlocksPerBatch = (sk + 127) / 128;
    info.numBlocks = batch * info.maxNumBlocksPerBatch;
    info.batch = batch;
    info.maxQSeqlen = sq;
    info.maxKvSeqlen = sk;
    info.qSeqlenList = qLengths.data();
    info.kvSeqlenList = kvLengths.data();
    info.scaleValue = scale;
    info.maskType = optiling::MaskType::MASK_SPEC;
    info.dataType = dtype == 0 ? optiling::DataType::FP16 : optiling::DataType::BF16;
    info.layout = "TND";
    info.cacheLayout = "nd";
    info.pagedCacheFlag = true;
    info.pagedShapeFlag = true;
    info.flashDecodeFlag = false;  // Original example's explicit setting.
    info.preToken = 2147483647;
    info.nextToken = 0;
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // Original host metadata has coreInfo[25]; retain its capacity.
    cores = std::min<uint32_t>(cores, 24);
    if (cores == 0 || (sizes[5] != 128 && sizes[5] != 256)) return 0;
    auto* data = static_cast<FAInferTilingData*>(host);
    new (data) FAInferTilingData{};
    optiling::FAInferTiling tiler(info);
    tiler.SetCoreNum(cores);
    tiler.DoTiling(*data);
    if (sizes[5] == 256) {
        // Tile dimensions only. Keep every native buffer count, prelaunch,
        // synchronization, instruction path and reduction loop unchanged.
        data->set_qkL1TileKLeft(256);
        data->set_qkL1TileKRight(256);
        data->set_qkL1TileN(128);
    }
    return std::max<uint64_t>(data->get_workSpaceSize(), 128ULL * 1024 * 1024);
}

extern "C" int classic_catlass70_run(void* q, void* k, void* v, void* mask,
    void* table, void* out, void* lse, void* qLengths, void* kvLengths,
    void* workspace, void* tiling, void* stream, int dtype) {
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    cores = std::min<uint32_t>(cores, 24);
#define CLASSIC_CATLASS70_LAUNCH(T) \
    FAInfer<T, float, Format::TND, Format::TND, CacheMode::pagedCache, \
            PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd> \
        <<<cores, nullptr, stream>>>( \
            static_cast<uint8_t*>(q), static_cast<uint8_t*>(k), static_cast<uint8_t*>(v), \
            static_cast<uint8_t*>(mask), static_cast<uint8_t*>(table), static_cast<uint8_t*>(out), \
            static_cast<uint8_t*>(lse), static_cast<uint8_t*>(qLengths), static_cast<uint8_t*>(kvLengths), \
            static_cast<uint8_t*>(workspace), static_cast<uint8_t*>(tiling))
    if (dtype == 0) { CLASSIC_CATLASS70_LAUNCH(half); }
    else if (dtype == 1) { CLASSIC_CATLASS70_LAUNCH(bfloat16_t); }
    else return -1;
#undef CLASSIC_CATLASS70_LAUNCH
    return 0;
}
