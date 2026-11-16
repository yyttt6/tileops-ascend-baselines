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
 * \file broadcast_sch_util_last_transpose.h
 * \brief
 */
#ifndef BROADCAST_SCH_UTILS_LAST_TRANSPOSE_H_
#define BROADCAST_SCH_UTILS_LAST_TRANSPOSE_H_

#include "atvoss/util/kernel_vec_intf_apt.h"

namespace Ops {
namespace Base {
static constexpr int64_t BROADCAST_MAX_DIMS_LAST_TRANSPOSE = 4;
static constexpr int64_t NDDMA_MAX_DIMS_LAST_TRANSPOSE = 4;
static constexpr int64_t BROADCAST_BROADCAST_DIMENTION_TWO_LAST_TRANSPOSE = 2;

template <std::size_t N>
__aicore__ inline int64_t BroadcastGetGmOffsetLastTranspose(const int64_t (&axesIndices)[N],
                                                            const int64_t (&totalStrides)[N], int64_t endIdx,
                                                            int64_t endDim, int64_t shapeLen, int64_t lastAxisLoopIdx,
                                                            int64_t lastDimFactor)
{
    int64_t gmOffset = 0;
    for (int64_t idx = 0; idx < endIdx; idx++) {
        gmOffset += axesIndices[idx] * totalStrides[idx];
    }
    gmOffset += axesIndices[endIdx] * totalStrides[endIdx] * endDim +
                lastAxisLoopIdx * totalStrides[shapeLen - 1] * lastDimFactor;
    return gmOffset;
}

template <typename T, std::size_t N>
__aicore__ inline int64_t BroadcastFuseAxesWithPad(const int64_t (&totalDims)[N], int64_t startIdx, int64_t endIdx)
{
    int64_t product = 1;
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        product *= totalDims[idx];
    }
    return product;
}

template <typename T, std::size_t N>
__aicore__ inline AscendC::NdDmaParams<T, NDDMA_MAX_DIMS_LAST_TRANSPOSE> BroadcastSetNddmaConfigWithoutLoop(
    const int64_t (&outputDims)[N], const int64_t (&outputStridesWithPad)[N], const int64_t (&inputStrides)[N],
    int64_t shapeLen, int64_t ubSplitAxis, int64_t ubSplitSize, int64_t lastDimFactor)
{
    AscendC::NdDmaLoopInfo<NDDMA_MAX_DIMS_LAST_TRANSPOSE> loopInfo;
    int64_t axisInsideUb = NDDMA_MAX_DIMS_LAST_TRANSPOSE - (shapeLen - ubSplitAxis);

    // 高位直接设置 loopSize为1
    for (int64_t i = 0; i < axisInsideUb; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = 1;
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = inputStrides[ubSplitAxis];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 -
                               i] = outputStridesWithPad[ubSplitAxis]; // outputStride 为ub内stride， -2轴补pad.
    }

    // 设置ub切分轴
    loopInfo.loopSize[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - axisInsideUb] = ubSplitSize;

    if (shapeLen - ubSplitAxis == BROADCAST_BROADCAST_DIMENTION_TWO_LAST_TRANSPOSE) {
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - axisInsideUb] = inputStrides[ubSplitAxis];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - axisInsideUb] = lastDimFactor;
    } else {
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - axisInsideUb] = inputStrides[ubSplitAxis];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - axisInsideUb] = outputStridesWithPad[ubSplitAxis];
    }

    for (uint64_t i = axisInsideUb + 1; i < NDDMA_MAX_DIMS_LAST_TRANSPOSE; i++) {
        if (i == NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1) {
            loopInfo.loopSize[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = lastDimFactor;
        } else {
            loopInfo.loopSize[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = outputDims[ubSplitAxis + i - axisInsideUb];
        }

        loopInfo.loopSrcStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = inputStrides[ubSplitAxis + i - axisInsideUb];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS_LAST_TRANSPOSE - 1 - i] = outputStridesWithPad[ubSplitAxis + i -
                                                                                             axisInsideUb];
    }

    T constValue = 0;
    AscendC::NdDmaParams<T, NDDMA_MAX_DIMS_LAST_TRANSPOSE> paramsMain = {loopInfo, constValue};
    return paramsMain;
}

template <typename T1, size_t N>
__aicore__ inline void MovAlign2UB(AscendC::LocalTensor<T1>& ubTensor, AscendC::GlobalTensor<T1>& inputGm,
                                   const int64_t (&outputDims)[N], const int64_t (&inputStrides)[N],
                                   const int64_t (&outputStridesWithPad)[N], int64_t ubSplitAxis, int64_t shapeLen,
                                   int64_t ubSplitSize, int64_t lastDimFactor)
{
    AscendC::DataCopyExtParams dataCopyExtParams;
    AscendC::DataCopyPadExtParams<T1> dataCopyPadExtParams;
    AscendC::LoopModeParams loopModeParams;

    dataCopyExtParams.blockLen = lastDimFactor * sizeof(T1);
    dataCopyExtParams.dstStride = 0;
    // tiling在合轴的时候，不会补维，所以这边有可能是3维 也可能是2维，由shapeLen决定，至少两维。
    // 同时ub切分轴，也可能在1维 也可能在0维。
    if (shapeLen - ubSplitAxis == BROADCAST_BROADCAST_DIMENTION_TWO_LAST_TRANSPOSE) {
        // [ubSplitSize, lastDimFactor]
        // [xxx, ubSplitSize, lastDimFactor]
        dataCopyExtParams.blockCount = ubSplitSize;
        dataCopyExtParams.srcStride = (inputStrides[ubSplitAxis] - lastDimFactor) * sizeof(T1);
        ;
        loopModeParams.loop1Size = 1;
        loopModeParams.loop1SrcStride = 0;
        loopModeParams.loop1DstStride = 0;
    } else {
        // [ubSplitSize, dim1, lastDimFactor]
        dataCopyExtParams.blockCount = outputDims[1];
        dataCopyExtParams.srcStride = (inputStrides[1] - lastDimFactor) * sizeof(T1);
        loopModeParams.loop1Size = ubSplitSize;
        loopModeParams.loop1SrcStride = inputStrides[0] * sizeof(T1);
        loopModeParams.loop1DstStride = outputStridesWithPad[0] * sizeof(T1);
    }

    loopModeParams.loop2Size = 1;
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop2DstStride = 0;

    AscendC::SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    AscendC::DataCopyPad<T1, PaddingMode::Compact>(ubTensor, inputGm, dataCopyExtParams, dataCopyPadExtParams);
    AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T1, size_t N>
__aicore__ inline void MovAlign2GM(AscendC::GlobalTensor<T1>& outputGm, AscendC::LocalTensor<T1>& ubTensor,
                                   const int64_t (&outputDims)[N], const int64_t (&outputStridesWithPad)[N],
                                   const int64_t (&outputStrides)[N], int64_t ubSplitAxis, int64_t shapeLen,
                                   int64_t ubSplitSize, int64_t lastDimFactor)
{
    AscendC::DataCopyExtParams dataCopyExtParams;
    AscendC::LoopModeParams loopModeParams;

    // ub切分在倒数第二维
    if (shapeLen - ubSplitAxis == BROADCAST_BROADCAST_DIMENTION_TWO_LAST_TRANSPOSE) {
        // [ubSplitSize, lastDimFactor]
        // [xxx, ubSplitSize, lastDimFactor]
        dataCopyExtParams.blockCount = ubSplitSize;
        dataCopyExtParams.dstStride = (outputStrides[ubSplitAxis] - lastDimFactor) * sizeof(T1);
        loopModeParams.loop1Size = 1;
        loopModeParams.loop1SrcStride = 0;
        loopModeParams.loop1DstStride = 0;
    } else {
        // [ubSplitSize, dim1, lastDimFactor]
        dataCopyExtParams.blockCount = outputDims[1];
        dataCopyExtParams.dstStride = 0;
        loopModeParams.loop1Size = ubSplitSize;
        loopModeParams.loop1SrcStride = outputStridesWithPad[0] * sizeof(T1);
        loopModeParams.loop1DstStride = outputStrides[0] * sizeof(T1);
    }

    dataCopyExtParams.blockLen = lastDimFactor * sizeof(T1);
    dataCopyExtParams.srcStride = 0;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop2DstStride = 0;

    AscendC::SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    AscendC::DataCopyPad<T1, PaddingMode::Compact>(outputGm, ubTensor, dataCopyExtParams);
    AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}
} // namespace Base
} // namespace Ops
#endif // BROADCAST_SCH_UTILS_LAST_TRANSPOSE_H_