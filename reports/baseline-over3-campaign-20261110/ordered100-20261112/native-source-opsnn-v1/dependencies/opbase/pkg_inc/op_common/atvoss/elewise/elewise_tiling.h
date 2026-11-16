/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elewise_tiling.h
 * \brief atvoss elewise template tiling
 */

#ifndef OP_COMMON_ATVOSS_ELEWISE_ELEWISE_TILING_H
#define OP_COMMON_ATVOSS_ELEWISE_ELEWISE_TILING_H

#include "register/tilingdata_base.h"
#include "elewise_base_struct.h"
#include "exe_graph/runtime/tiling_context.h"
#include "op_common/log/log.h"

namespace Ops {
namespace Base {

using namespace optiling;

constexpr uint64_t OP_KEY_OFFSET = 100000000000000;
constexpr int64_t BITS_NUM = 8;
constexpr int64_t DEFAULT_SCHEMODE = 1;

enum class OPBASE_API BITS_SIZE {
    BITS1_SIZE = 1,
    BITS8_SIZE = 8,
    BITS16_SIZE = 16,
    BITS32_SIZE = 32,
    BITS64_SIZE = 64
};

struct OPBASE_API ComputeParams {
    int64_t maxDtypeBits;
    int64_t minDtypeBits;
    std::vector<int64_t> extraSize;
    std::vector<int64_t> bufferDivisor;
};

struct OPBASE_API ElewiseTilingParams {
    int64_t coreNum;
    gert::Shape shape;
    int64_t ubSize;
    std::map<uint64_t, ComputeParams> computeMap;
};

struct OPBASE_API ElewiseCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct OPBASE_API ElewiseTilingData {
    uint64_t innerKey;
    int64_t dim0;
    int64_t ubFormer;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t coreNum;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
};

class OPBASE_API ElewiseBaseTiling {
public:
    explicit ElewiseBaseTiling(gert::TilingContext* context) : context_(context) {}

    template <typename OpDag, bool CheckShape = true>
    ge::graphStatus DoTiling(EleBaseTilingData& eleTilingData, int64_t extraSize = 0, int64_t extraBufferNum = 0);
    template <typename OpDag, bool CheckShape>
    ge::graphStatus DoTiling(ElewiseTilingData& elewiseTilingData, int64_t extraSize = 0, int64_t extraBufferNum = 0);
    template <typename OpDag, bool CheckShape = true>
    ge::graphStatus DoTiling(EleBaseTilingData16B& eleTilingData16B, int64_t extraSize = 0, int64_t extraBufferNum = 0);
    template <typename OpDag, bool CheckShape = true>
    ge::graphStatus DoTiling24B(int64_t extraSize = 0, int64_t extraBufferNum = 0);
    template <typename OpDag, bool CheckShape = true>
    ge::graphStatus DoTiling32B(int64_t extraSize = 0, int64_t extraBufferNum = 0);

    // 4bits
    template <typename OpDag, bool CheckShape = true>
    ge::graphStatus DoTiling4Bits(EleBaseTilingData& eleTilingData, int64_t extraSize = 0, int64_t extraBufferNum = 0);
    template <typename OpDag, bool CheckShape>
    ge::graphStatus DoTiling4Bits(ElewiseTilingData& elewiseTilingData, int64_t extraSize = 0,
                                  int64_t extraBufferNum = 0);

    template <typename T>
    void SetScalar(T value);
    int64_t GetBlockDim() const;
    int64_t GetNumBlocks();

private:
    int64_t GetScheMode() const;
    ge::graphStatus GetPlatformInfo();
    template <bool CheckShape = true>
    ge::graphStatus GetShapeInfo();
    void AdaptEleBaseTilingData(const ElewiseTilingData& elewiseTilingData, EleBaseTilingData& eleBaseTilingData) const;
    void AdaptEleBaseTilingData16B(const ElewiseTilingData& elewiseTilingData,
                                   EleBaseTilingData16B& eleBaseTilingData16B);
    ge::graphStatus AdaptEleBaseTilingData24B(const ElewiseTilingData& elewiseTilingData);
    ge::graphStatus AdaptEleBaseTilingData32B(const ElewiseTilingData& elewiseTilingData);
    EleBaseTilingData32B* Ele32BTilingData = nullptr;
    EleBaseTilingData24B* Ele24BTilingData = nullptr;
    char scalarData[ELEWISE_MAX_SCALAR_BYTES] = {0};
    gert::TilingContext* context_{nullptr};
    uint32_t offset = 0;
    int64_t numBlocks;
    uint64_t coreNum{0};
    uint64_t ubSize{0};
    gert::Shape shape;
};

OPBASE_API ge::graphStatus ElewiseTiling(const ElewiseTilingParams& elewiseTilingParams,
                                         ElewiseTilingData& elewiseTilingData);
OPBASE_API bool IsSameElewiseShape(const gert::Shape&, const gert::Shape&);

template <bool CheckShape>
ge::graphStatus ElewiseBaseTiling::GetShapeInfo()
{
    auto outputShape = context_->GetOutputShape(0);
    if (outputShape == nullptr) {
        OP_LOGW(context_, "will ignore output shape check");
        for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
            auto inputShape = context_->GetInputShape(i);
            if (inputShape == nullptr) {
                OP_LOGI(context_, "Input[%lu] is nullptr, will ignore set max elewise shape.", i);
                continue;
            }

            shape = inputShape->GetStorageShape();
            if (!(shape.IsScalar() || shape.GetShapeSize() == 1)) {
                OP_LOGI(context_, "Input[%lu] is not scalar, will set max elewise shape", i);
                break;
            }
        }
    } else {
        shape = outputShape->GetStorageShape();
    }
    if (!CheckShape) {
        return ge::GRAPH_SUCCESS;
    }
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto inputShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        auto storageShape = inputShape->GetStorageShape();
        if (!IsSameElewiseShape(shape, storageShape)) {
            OP_LOGE(context_, "Input[%lu] shape is not equal with output shape", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 0.5字宽的数据，如fp4
template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling4Bits(ElewiseTilingData& elewiseTilingData, int64_t extraSize,
                                                 int64_t extraBufferNum)
{
    auto status = GetPlatformInfo();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Get platform info failed.");
        return ge::GRAPH_FAILED;
    }
    status = GetShapeInfo<CheckShape>();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Get shape info faild.");
        return ge::GRAPH_FAILED;
    }
    ComputeParams params;
    params.maxDtypeBits = OpDag::MaxDtypeBytes * BITS_NUM;
    params.minDtypeBits = OpDag::MinDtypeBytes * BITS_NUM;
    params.extraSize = {extraSize};
    params.bufferDivisor = {(OpDag::BufferNum + extraBufferNum) * params.maxDtypeBits};

    // 合轴
    int64_t dim0 = 1;
    for (uint64_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) == 0, OP_LOGE("ElewiseBaseTiling::DoTiling4Bits", "elewiseTiling dim can not be 0"),
                    return ge::GRAPH_FAILED);
        dim0 = dim0 * shape.GetDim(i);
    }
    // 两个fp4类型数据当做 一个int8类型数据看待，向上取整
    dim0 = (dim0 + 1) / 2;

    ElewiseTilingParams elewiseTilingParams;
    elewiseTilingParams.shape = gert::Shape({dim0});
    elewiseTilingParams.coreNum = coreNum;
    elewiseTilingParams.ubSize = ubSize;
    elewiseTilingParams.computeMap = {{0, params}};
    status = ElewiseTiling(elewiseTilingParams, elewiseTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 0.5字宽的数据，如fp4
template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling4Bits(EleBaseTilingData& eleTilingData, int64_t extraSize,
                                                 int64_t extraBufferNum)
{
    ElewiseTilingData elewiseTilingData;
    auto status = DoTiling4Bits<OpDag, CheckShape>(elewiseTilingData, extraSize, extraBufferNum);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    AdaptEleBaseTilingData(elewiseTilingData, eleTilingData);
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling(ElewiseTilingData& elewiseTilingData, int64_t extraSize,
                                            int64_t extraBufferNum)
{
    auto status = GetPlatformInfo();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Get platform info failed.");
        return ge::GRAPH_FAILED;
    }
    status = GetShapeInfo<CheckShape>();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Get shape info faild.");
        return ge::GRAPH_FAILED;
    }
    ComputeParams params;
    params.maxDtypeBits = OpDag::MaxDtypeBytes * BITS_NUM;
    params.minDtypeBits = OpDag::MinDtypeBytes * BITS_NUM;
    params.extraSize = {extraSize};
    params.bufferDivisor = {(OpDag::BufferNum + extraBufferNum) * params.maxDtypeBits};

    ElewiseTilingParams elewiseTilingParams;
    elewiseTilingParams.shape = shape;
    elewiseTilingParams.coreNum = coreNum;
    elewiseTilingParams.ubSize = ubSize;
    elewiseTilingParams.computeMap = {{0, params}};
    status = ElewiseTiling(elewiseTilingParams, elewiseTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling(EleBaseTilingData& eleTilingData, int64_t extraSize, int64_t extraBufferNum)
{
    ElewiseTilingData elewiseTilingData;
    auto status = DoTiling<OpDag, CheckShape>(elewiseTilingData, extraSize, extraBufferNum);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    AdaptEleBaseTilingData(elewiseTilingData, eleTilingData);
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling(EleBaseTilingData16B& eleTilingData16B, int64_t extraSize,
                                            int64_t extraBufferNum)
{
    ElewiseTilingData elewiseTilingData;
    auto status = DoTiling<OpDag, CheckShape>(elewiseTilingData, extraSize, extraBufferNum);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    AdaptEleBaseTilingData16B(elewiseTilingData, eleTilingData16B);
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling24B(int64_t extraSize, int64_t extraBufferNum)
{
    ElewiseTilingData elewiseTilingData;
    auto status = DoTiling<OpDag, CheckShape>(elewiseTilingData, extraSize, extraBufferNum);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return AdaptEleBaseTilingData24B(elewiseTilingData);
}

template <typename OpDag, bool CheckShape>
ge::graphStatus ElewiseBaseTiling::DoTiling32B(int64_t extraSize, int64_t extraBufferNum)
{
    ElewiseTilingData elewiseTilingData;
    auto status = DoTiling<OpDag, CheckShape>(elewiseTilingData, extraSize, extraBufferNum);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return AdaptEleBaseTilingData32B(elewiseTilingData);
}

template <typename T>
void ElewiseBaseTiling::SetScalar(T value)
{
    if (Ele24BTilingData != nullptr) {
        *reinterpret_cast<T*>(&Ele24BTilingData->scalarData[offset]) = value;
    } else if (Ele32BTilingData != nullptr) {
        *reinterpret_cast<T*>(&Ele32BTilingData->scalarData[offset]) = value;
    } else {
        *reinterpret_cast<T*>(&scalarData[offset]) = value;
    }
    offset += sizeof(T);
}

} // namespace Base
} // namespace Ops
#endif // OP_COMMON_ATVOSS_ELEWISE_ELEWISE_TILING_H
