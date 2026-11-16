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
 * \file broadcast_tiling_noncontiguous.h
 * \brief atvoss broadcast template tiling
 */
#ifndef BROADCAST_TILING_NONCONTIGUOUS_H_
#define BROADCAST_TILING_NONCONTIGUOUS_H_
#include <type_traits>
#include "broadcast_tiling_base.h"

namespace Ops {
namespace Base {
OPBASE_API uint64_t GetBlockSplitFactorLastTranspose(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                                     uint64_t maxElemNum, const BroadcastComputeParams& computeParams);

OPBASE_API ge::graphStatus DoBrodcastTilingLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                                         BroadcastTilingData& broadcastTilingData);

OPBASE_API uint64_t GetBlockSplitFactorNLastTranspose(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                                      uint64_t maxElemNum, const BroadcastComputeParams& computeParams,
                                                      bool isUbBroadcast);

OPBASE_API ge::graphStatus BroadcastTilingNLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                                         BroadcastTilingData& broadcastTilingData, bool isUbBroadcast);

/**
 *  按顺序将对应节点的输入shape信息设置到tilingdata中
 * @tparam pos 按顺序执行递归的位置
 * @return
 */
template <typename OpDag, typename T, int pos = 0>
void SetInputNddmaParams(T* brcLastTransposeTilingData, BroadcastTilingData& tilingData)
{
    if constexpr (OpDag::CopyBrcSize != 0) {
        using Op = typename OpDag::CopyBrcNodes::template At<pos>;
        using Input = typename Op::InHolders::template At<0>;
        int64_t idx = Input::Pos;
        std::copy(tilingData.dims[idx].begin(), tilingData.dims[idx].end(),
                  *(brcLastTransposeTilingData->inputBrcDims + pos));
        std::copy(tilingData.strides[idx].begin(), tilingData.strides[idx].end(),
                  *(brcLastTransposeTilingData->inputBrcStrides + pos));
        if constexpr (pos + 1 < OpDag::CopyBrcSize) {
            SetInputNddmaParams<OpDag, T, pos + 1>(brcLastTransposeTilingData, tilingData);
        }
    }
}

/**
 *  获取copyInBrc的位置
 * @tparam pos 按顺序执行递归的位置
 * @return
 */
template <typename OpDag, int pos = 0>
void GetCopyInBrcPosList(std::set<int64_t>& copyInBrcPos)
{
    if constexpr (OpDag::CopyBrcSize != 0) {
        using Op = typename OpDag::CopyBrcNodes::template At<pos>;
        using Input = typename Op::InHolders::template At<0>;
        int64_t idx = Input::Pos;
        copyInBrcPos.insert(idx);
        if constexpr (pos + 1 < OpDag::CopyBrcSize) {
            GetCopyInBrcPosList<OpDag, pos + 1>(copyInBrcPos);
        }
    }
}

/**
 *  获取VecBrc的位置，并按顺序将对应节点的输入shape信息设置到tilingdata中
 *  @tparam pos 按顺序执行递归的位置
 */
template <typename OpDag, typename T, int pos = 0>
void SetInputUbBrcParams(T* brcLastTransposeTilingData, BroadcastTilingData& tilingData)
{
    if constexpr (OpDag::VecBrcSize > 0) {
        using Op = typename OpDag::VecBrcNodes::template At<pos>;
        using Input = typename Op::SourceInHolders::template At<0>;
        int64_t idx = Input::Pos;

        std::copy(tilingData.dims[idx].begin(), tilingData.dims[idx].end(),
                  *(brcLastTransposeTilingData->inputVecBrcDims + pos));

        brcLastTransposeTilingData->inputVecBrcStrides[pos] = tilingData.strides[idx][tilingData.ubSplitAxis];
        if constexpr (pos + 1 < OpDag::VecBrcSize) {
            SetInputUbBrcParams<OpDag, T, pos + 1>(brcLastTransposeTilingData, tilingData);
        }
    }
}

/**
 *  根据切分结果，设置CopyIn节点对应的切分大小
 * @return
 */
template <typename OpDag, typename T>
void SetInputParams(T* brcLastTransposeTilingData, BroadcastTilingData& tilingData, std::set<int64_t>& copyInBrcPos)
{
    for (uint32_t i = 0; i < OpDag::InputSize; i++) {
        if (copyInBrcPos.count(i)) {
            continue;
        }
        std::copy(tilingData.strides[i].begin(), tilingData.strides[i].end(),
                  *(brcLastTransposeTilingData->inputStrides + i));
        std::copy(tilingData.dims[i].begin(), tilingData.dims[i].end(), *(brcLastTransposeTilingData->inputDims + i));
    }
}

/**
 *  适配tilingData
 * @return
 */
template <typename OpDag, typename T>
void AdaptBroadcastLastTransposeTilingData(T* brcLastTransposeTilingData, BroadcastTilingData& tilingData)
{
    brcLastTransposeTilingData->blockFormer = tilingData.blockFormer;
    brcLastTransposeTilingData->ubFormer = tilingData.ubFormer;
    brcLastTransposeTilingData->ubOuter = tilingData.ubOuter;
    brcLastTransposeTilingData->ubTail = tilingData.ubTail;
    brcLastTransposeTilingData->ubFormerLastAxis = tilingData.ubFormerLastAxis;
    brcLastTransposeTilingData->ubOuterLastAxis = tilingData.ubOuterLastAxis;
    brcLastTransposeTilingData->ubTailLastAxis = tilingData.ubTailLastAxis;

    brcLastTransposeTilingData->blockTail = tilingData.blockTail;
    brcLastTransposeTilingData->shapeLen = tilingData.shapeLen;
    brcLastTransposeTilingData->ubSplitAxis = tilingData.ubSplitAxis;
    brcLastTransposeTilingData->dimProductBeforeUbInner = tilingData.dimProductBeforeUbInner;
    brcLastTransposeTilingData->elemNum = tilingData.elemNum;
    brcLastTransposeTilingData->blockNum = tilingData.blockNum;
    brcLastTransposeTilingData->minDtypeBlockAlignSize = tilingData.minDtypeBlockAlignSize;
    std::copy(tilingData.dims.back().begin(), tilingData.dims.back().end(), brcLastTransposeTilingData->outputDims);
    std::copy(tilingData.strides.back().begin(), tilingData.strides.back().end(),
              brcLastTransposeTilingData->outputStrides);

    // 逐个拷贝生成outputStridesWithPad
    for (int64_t i = tilingData.shapeLen - 1; i >= 0; i--) {
        if (i > tilingData.shapeLen - 3) {
            brcLastTransposeTilingData->outputStridesWithPad[i] = brcLastTransposeTilingData->outputStrides[i];
        } else if (i == tilingData.shapeLen - 3) {
            int64_t oriStride = brcLastTransposeTilingData->outputStrides[i];
            int64_t oriStrideAlign = (oriStride + tilingData.minDtypeBlockAlignSize - 1) /
                                     tilingData.minDtypeBlockAlignSize * tilingData.minDtypeBlockAlignSize;
            brcLastTransposeTilingData->outputStridesWithPad[i] = oriStrideAlign;
        } else {
            brcLastTransposeTilingData->outputStridesWithPad[i] = brcLastTransposeTilingData
                                                                      ->outputStridesWithPad[i + 1] *
                                                                  brcLastTransposeTilingData->outputDims[i + 1];
        }
    }
}

template <typename OpDag, typename T>
ge::graphStatus DoBroadcastTilingLastTranspose(BroadcastTilingParams& broadcastTilingParams,
                                               T* brcLastTransposeTilingData, BroadcastTilingData& tilingData)
{
    auto status = DoBrodcastTilingLastTranspose(broadcastTilingParams, tilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "inner DoBroadcastOpTilingLastTranspose tiling failed");
        return ge::GRAPH_FAILED;
    }

    std::set<int64_t> copyInBrcPos;
    GetCopyInBrcPosList<OpDag>(copyInBrcPos);

    // adapt
    AdaptBroadcastLastTransposeTilingData<T>(brcLastTransposeTilingData, tilingData);
    SetInputNddmaParams<OpDag, T>(brcLastTransposeTilingData, tilingData);
    SetInputUbBrcParams<OpDag, T>(brcLastTransposeTilingData, tilingData);
    SetInputParams<OpDag, T>(brcLastTransposeTilingData, tilingData, copyInBrcPos);

    return ge::GRAPH_SUCCESS;
}
template <typename OpDag, typename T>
void AdaptBroadcastNLastTransposeTilingData(T* brcNLastTransposeTilingData, BroadcastTilingData& tilingData,
                                            bool isUbBroadcast)
{
    brcNLastTransposeTilingData->blockFormer = tilingData.blockFormer;
    brcNLastTransposeTilingData->ubFormer = tilingData.ubFormer;
    brcNLastTransposeTilingData->ubOuter = tilingData.ubOuter;
    brcNLastTransposeTilingData->ubTail = tilingData.ubTail;
    brcNLastTransposeTilingData->blockTail = tilingData.blockTail;
    brcNLastTransposeTilingData->shapeLen = tilingData.shapeLen;
    brcNLastTransposeTilingData->ubSplitAxis = tilingData.ubSplitAxis;
    brcNLastTransposeTilingData->dimProductBeforeUbInner = tilingData.dimProductBeforeUbInner;
    brcNLastTransposeTilingData->elemNum = tilingData.elemNum;
    brcNLastTransposeTilingData->blockNum = tilingData.blockNum;
    brcNLastTransposeTilingData->minDtypeBlockAlignSize = tilingData.minDtypeBlockAlignSize;
    std::copy(tilingData.dims.back().begin(), tilingData.dims.back().end(), brcNLastTransposeTilingData->outputDims);
    std::copy(tilingData.strides.back().begin(), tilingData.strides.back().end(),
              brcNLastTransposeTilingData->outputStrides);

    int64_t boundaryNum = 3;
    if (isUbBroadcast) {
        boundaryNum = 2;
    }

    // 逐个拷贝生成outputStridesWithPad
    for (int64_t i = tilingData.shapeLen - 1; i >= 0; i--) {
        if (i > tilingData.shapeLen - boundaryNum) {
            brcNLastTransposeTilingData->outputStridesWithPad[i] = brcNLastTransposeTilingData->outputStrides[i];
        } else if (i == tilingData.shapeLen - boundaryNum) {
            int64_t oriStride = brcNLastTransposeTilingData->outputStrides[i];
            int64_t oriStrideAlign = (oriStride + tilingData.minDtypeBlockAlignSize - 1) /
                                     tilingData.minDtypeBlockAlignSize * tilingData.minDtypeBlockAlignSize;
            brcNLastTransposeTilingData->outputStridesWithPad[i] = oriStrideAlign;
        } else {
            brcNLastTransposeTilingData->outputStridesWithPad[i] = brcNLastTransposeTilingData
                                                                       ->outputStridesWithPad[i + 1] *
                                                                   brcNLastTransposeTilingData->outputDims[i + 1];
        }
    }
}

template <typename OpDag, typename T>
ge::graphStatus DoBroadcastTilingNLastTranspose(BroadcastTilingParams& broadcastTilingParams,
                                                T* brcNLastTransposeTilingData, BroadcastTilingData& tilingData,
                                                bool isUbBroadcast)
{
    auto status = BroadcastTilingNLastTranspose(broadcastTilingParams, tilingData, isUbBroadcast);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "inner DoBroadcastOpTilingLastTranspose tiling failed");
        return ge::GRAPH_FAILED;
    }

    std::set<int64_t> copyInBrcPos;
    GetCopyInBrcPosList<OpDag>(copyInBrcPos);

    // adapt
    AdaptBroadcastNLastTransposeTilingData<T>(brcNLastTransposeTilingData, tilingData, isUbBroadcast);
    SetInputNddmaParams<OpDag, T>(brcNLastTransposeTilingData, tilingData);
    SetInputUbBrcParams<OpDag, T>(brcNLastTransposeTilingData, tilingData);
    SetInputParams<OpDag, T>(brcNLastTransposeTilingData, tilingData, copyInBrcPos);

    return ge::GRAPH_SUCCESS;
}
} // namespace Base
} // namespace Ops
#endif // BROADCAST_TILING_NONCONTIGUOUS_H_
