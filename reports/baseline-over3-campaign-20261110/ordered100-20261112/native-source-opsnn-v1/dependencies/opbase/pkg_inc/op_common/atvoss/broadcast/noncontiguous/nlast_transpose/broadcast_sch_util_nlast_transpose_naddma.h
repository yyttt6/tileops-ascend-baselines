/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file broadcast_sch_util_nlast_transpose_naddma.h
 * \brief
 */
#ifndef BROADCAST_SCH_UTIL_NLAST_TRANSPOSE_NADDMA_H_
#define BROADCAST_SCH_UTIL_NLAST_TRANSPOSE_NADDMA_H_

#include "atvoss/util/kernel_vec_intf_apt.h"

namespace Ops {
namespace Base {
static constexpr int64_t NEGATIVE_THREE = -3;
static constexpr int64_t NEGATIVE_TWO = -2;
static constexpr int64_t NEGATIVE_ONE = -1;
static constexpr int64_t BLOCK = 32;

template <std::size_t N>
__aicore__ inline void BroadcastGetAxesIndicesNDDMA(int64_t (&axesIndices)[N], int64_t prodIdx,
                                                    const int64_t (&totalDims)[N], int64_t endIdx, int64_t totalProduct)
{
    for (int64_t idx = 0; idx < endIdx; idx++) {
        totalProduct = totalProduct / totalDims[idx];
        int64_t curIdx = prodIdx / totalProduct;
        axesIndices[idx] = curIdx;
        prodIdx = prodIdx - curIdx * totalProduct;
    }
    axesIndices[endIdx] = prodIdx;

    return;
}

template <std::size_t N>
__aicore__ inline void BroadcastUpdateAxesIndicesNDDMA(int64_t (&axesIndices)[N], const int64_t (&totalDims)[N],
                                                       int64_t endIdx, int64_t endDim)
{
    axesIndices[endIdx]++;
    if (axesIndices[endIdx] == endDim) {
        axesIndices[endIdx] = 0;
        axesIndices[endIdx - 1]++;
    }
    for (int64_t idx = endIdx - 1; idx >= 0; idx--) {
        if (axesIndices[idx] == totalDims[idx]) {
            axesIndices[idx] = 0;
            axesIndices[idx - 1]++;
        }
    }

    return;
}

template <std::size_t N>
__aicore__ inline int64_t BroadcastGetGmOffsetNDDMA(const int64_t (&axesIndices)[N], const int64_t (&totalStrides)[N],
                                                    int64_t endIdx, int64_t endDim)
{
    int64_t gmOffset = 0;
    for (int64_t idx = 0; idx < endIdx; idx++) {
        gmOffset += axesIndices[idx] * totalStrides[idx];
    }
    gmOffset += axesIndices[endIdx] * totalStrides[endIdx] * endDim;
    return gmOffset;
}

template <size_t N>
__aicore__ inline int64_t CalAfterUbSplitAxisLen(const int64_t (&Dim)[N], int64_t shapeLen, int64_t ubSplitAxis)
{
    int64_t totalLen = 0;
    for (int64_t i = ubSplitAxis + 1; i < shapeLen; i++) {
        if (totalLen == 0) {
            totalLen = Dim[i];
        } else {
            totalLen *= Dim[i];
        }
    }
    return totalLen;
}

template <typename T, std::size_t N>
__aicore__ inline AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> BroadcastSetNlastTransposeNddmaConfigWithoutLoop(
    const int64_t (&outputDims)[N], const int64_t (&outputStrides)[N], const int64_t (&outputStridesWithPad)[N],
    const int64_t (&inputStrides)[N], int64_t shapeLen, int64_t ubSplitSize, int64_t ubSplitAxis)
{
    AscendC::NdDmaLoopInfo<NDDMA_MAX_DIMS> loopInfo;
    int64_t axisInsideUb = NDDMA_MAX_DIMS - (shapeLen - ubSplitAxis);
    for (int64_t i = 0; i < axisInsideUb; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - i] = 1;
        loopInfo.loopLpSize[NDDMA_MAX_DIMS - 1 - i] = 0;
        loopInfo.loopRpSize[NDDMA_MAX_DIMS - 1 - i] = 0;
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - i] = inputStrides[ubSplitAxis];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - i] = outputStridesWithPad[ubSplitAxis];
    }

    loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - axisInsideUb] = ubSplitSize;
    loopInfo.loopLpSize[NDDMA_MAX_DIMS - 1 - axisInsideUb] = 0;
    loopInfo.loopRpSize[NDDMA_MAX_DIMS - 1 - axisInsideUb] = 0;
    loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - axisInsideUb] = inputStrides[ubSplitAxis];
    loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - axisInsideUb] = outputStridesWithPad[ubSplitAxis];

    for (uint64_t i = axisInsideUb + 1; i < NDDMA_MAX_DIMS; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - i] = outputDims[ubSplitAxis + i - axisInsideUb];
        loopInfo.loopLpSize[NDDMA_MAX_DIMS - 1 - i] = 0;
        loopInfo.loopRpSize[NDDMA_MAX_DIMS - 1 - i] = 0;
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - i] = inputStrides[ubSplitAxis + i - axisInsideUb];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - i] = outputStridesWithPad[ubSplitAxis + i - axisInsideUb];
    }
    T constValue = 0;
    AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> paramsMain = {loopInfo, constValue};
    return paramsMain;
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void MovAlign2UBNlastNDDMA(AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor,
                                             const int64_t (&outputDims)[N], const int64_t (&outputStridesWithPad)[N],
                                             const int64_t (&inputStrides)[N], const int64_t (&inputDim)[N],
                                             int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize,
                                             int64_t gmOffset)
{
    // move align
    int64_t ubInnerDims[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {1};
    for (int64_t i = 0; i < shapeLen; i++) {
        if (i == ubSplitAxis) {
            ubInnerDims[i] = ubSplitSize;
        } else {
            ubInnerDims[i] = outputDims[i];
        }
    }
    AscendC::DataCopyExtParams dataCopyExtParams;
    AscendC::DataCopyPadExtParams<T1> dataCopyPadExtParams;
    AscendC::LoopModeParams loopParam2Ub;
    loopParam2Ub.loop1Size = 1;
    loopParam2Ub.loop2Size = 1;
    loopParam2Ub.loop1SrcStride = 0;
    loopParam2Ub.loop1DstStride = 0;
    loopParam2Ub.loop2SrcStride = 0;
    loopParam2Ub.loop2DstStride = 0;
    if (ubSplitAxis == shapeLen - 1) {
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * sizeof(T1);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
    } else {
        dataCopyExtParams.blockCount = ubInnerDims[shapeLen + NEGATIVE_TWO];
        dataCopyExtParams.blockLen = ubInnerDims[shapeLen + NEGATIVE_ONE] * sizeof(T1);
        dataCopyExtParams.srcStride = (inputStrides[shapeLen + NEGATIVE_TWO] - inputDim[shapeLen + NEGATIVE_ONE]) *
                                      sizeof(T1);
        dataCopyExtParams.dstStride = 0;
        // 如果shapeLen >= 3 则使用LoopModeParams循环
        if (shapeLen - ubSplitAxis >= 3) {
            loopParam2Ub.loop1Size = ubInnerDims[shapeLen + NEGATIVE_THREE];
            loopParam2Ub.loop1SrcStride = inputStrides[shapeLen + NEGATIVE_THREE] * sizeof(T1);
            loopParam2Ub.loop1DstStride = outputStridesWithPad[shapeLen + NEGATIVE_THREE] * sizeof(T1);
            if (shapeLen - ubSplitAxis >= 4) {
                loopParam2Ub.loop2Size = ubInnerDims[0];
                loopParam2Ub.loop2SrcStride = inputStrides[0] * sizeof(T1);
                loopParam2Ub.loop2DstStride = outputStridesWithPad[0] * sizeof(T1);
            }
        }
    }
    AscendC::SetLoopModePara(loopParam2Ub, DataCopyMVType::OUT_TO_UB);
    if constexpr (AscendC::IsSameType<T1, T2>::value) {
        AscendC::DataCopyPad<T1, PaddingMode::Compact>(outputTensor, inputGm[gmOffset], dataCopyExtParams,
                                                       dataCopyPadExtParams);
    } else {
        AscendC::DataCopyPad<T1, PaddingMode::Compact>(outputTensor.template ReinterpretCast<T1>(), inputGm[gmOffset],
                                                       dataCopyExtParams, dataCopyPadExtParams);
    }
    AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T1, size_t N>
__aicore__ inline void MovAlign2GMNlastNDDMA(AscendC::GlobalTensor<T1>& outputGm, AscendC::LocalTensor<T1>& ubTensor,
                                             const int64_t (&outputDims)[N], const int64_t (&outputStrides)[N],
                                             const int64_t (&outputStridesWithPad)[N], int64_t shapeLen,
                                             int64_t ubSplitAxis, int64_t gmOffset, int64_t ubSplitSize)
{
    int64_t ubInnerDims[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {1};
    for (int64_t i = 0; i < shapeLen; i++) {
        if (i == ubSplitAxis) {
            ubInnerDims[i] = ubSplitSize;
        } else {
            ubInnerDims[i] = outputDims[i];
        }
    }
    AscendC::LoopModeParams loopParam2GM;
    loopParam2GM.loop1Size = 1;
    loopParam2GM.loop2Size = 1;
    loopParam2GM.loop1SrcStride = 0;
    loopParam2GM.loop1DstStride = 0;
    loopParam2GM.loop2SrcStride = 0;
    loopParam2GM.loop2DstStride = 0;
    AscendC::DataCopyExtParams dataCopyExtParams;
    if (ubSplitAxis == shapeLen - 1) {
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * sizeof(T1);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
    } else {
        dataCopyExtParams.blockCount = ubInnerDims[shapeLen + NEGATIVE_TWO];
        dataCopyExtParams.blockLen = ubInnerDims[shapeLen + NEGATIVE_ONE] * sizeof(T1);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
        // 如果shapeLen >= 3 则使用LoopModeParams循环
        if (shapeLen - ubSplitAxis >= 3) {
            loopParam2GM.loop1Size = ubInnerDims[shapeLen + NEGATIVE_THREE];
            loopParam2GM.loop1SrcStride = outputStridesWithPad[shapeLen + NEGATIVE_THREE] * sizeof(T1);
            loopParam2GM.loop1DstStride = dataCopyExtParams.blockCount * dataCopyExtParams.blockLen;
            if (shapeLen - ubSplitAxis >= 4) {
                loopParam2GM.loop2Size = ubInnerDims[0];
                loopParam2GM.loop2SrcStride = outputStridesWithPad[0] * sizeof(T1);
                loopParam2GM.loop2DstStride = outputStrides[0] * sizeof(T1);
            }
        }
    }

    AscendC::SetLoopModePara(loopParam2GM, DataCopyMVType::UB_TO_OUT);
    AscendC::DataCopyPad<T1, PaddingMode::Compact>(outputGm, ubTensor, dataCopyExtParams);
    AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNlastTransposeNddmaWithoutLoop(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&outputStridesWithPad)[N], const int64_t (&inputStrides)[N],
    const int64_t (&axesIndices)[N], const int64_t (&inputDim)[N], int64_t ubSplitAxis, int64_t shapeLen,
    int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffsetNDDMA(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    // 如果ubSplitAxis做brc 或者ub切分轴后的inputShape乘积和outputStride[ubSplitAxis]不等则做NDDMA，否则做move align
    if (inputStrides[ubSplitAxis] == 0 ||
        (outputStrides[ubSplitAxis] != CalAfterUbSplitAxisLen(inputDim, shapeLen, ubSplitAxis))) {
        // NDDMA
        static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
        AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNlastTransposeNddmaConfigWithoutLoop<T1>(
            outputDims, outputStrides, outputStridesWithPad, inputStrides, shapeLen, ubSplitSize, ubSplitAxis);
        if constexpr (AscendC::IsSameType<T1, T2>::value) {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor, inputGm[gmOffset], paramsMain);
        } else {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor.template ReinterpretCast<T1>(),
                                                          inputGm[gmOffset], paramsMain);
        }
    } else {
        MovAlign2UBNlastNDDMA(inputGm, outputTensor, outputDims, outputStridesWithPad, inputStrides, inputDim,
                              ubSplitAxis, shapeLen, ubSplitSize, gmOffset);
    }
}
} // namespace Base
} // namespace Ops

#endif // BROADCAST_SCH_UTIL_NLAST_TRANSPOSE_NADDMA_H_