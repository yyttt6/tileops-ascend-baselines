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
 * \file broadcast_utils_transpose.h
 * \brief
 */
#ifndef BROADCAST_UTILS_TRANSPOSE_H_
#define BROADCAST_UTILS_TRANSPOSE_H_

namespace Ops {
namespace Base {
template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNddmaNonContiguousWithoutLoop(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    static constexpr AscendC::NdDmaConfig config = {
        false, 0, 0, false}; // 设定 config参数，用于指定brc操作的pad填充具体值，设置后就无须设置Lp和Rp
    AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNddmaConfigWithoutLoop<T1>(
        outputDims, outputStrides, inputStrides, shapeLen, ubSplitSize,
        ubSplitAxis); // 设定用于DataCopy搬运的param，主要包含loopSize(每个dim具体处理的元素数量)、loopSrcStride和loopDstStride,后面两个中包含brc和tra信息
    if constexpr (AscendC::IsSameType<T1, T2>::value) {
        AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor, inputGm[gmOffset], paramsMain);
    } else {
        AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor.template ReinterpretCast<T1>(), inputGm[gmOffset],
                                                      paramsMain);
    }
}

template <typename T1, typename T2, size_t N>
__aicore__ __attribute__((noinline)) void BroadcastNddmaNonContiguousWithoutLoopNoInline(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    BroadcastNddmaNonContiguousWithoutLoop(inputGm, outputTensor, outputDims, outputStrides, inputStrides, axesIndices,
                                           ubSplitAxis, shapeLen, ubSplitSize, ubFormer);
}

// withloop 即ub切分后nddma的维度已经超过了5维
template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNddmaNonContiguousWithLoop(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
    AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNddmaConfigWithLoop<T1>(
        outputDims, outputStrides, inputStrides, shapeLen, ubSplitAxis);
    int64_t nddmaIndices[NDDMA_THROW_DIMS] = {0};
    int64_t nddmaProduct = BroadcastFuseAxes(outputDims, ubSplitAxis + 1, shapeLen - NDDMA_MAX_DIMS) *
                           ubSplitSize; // nddma循环次数，即ub切分轴后一位连乘到nddma前一位
    for (int64_t i = 0; i < nddmaProduct; i++) {
        if (i != 0) {
            BroadcastUpdateNddmaAxesIndices(nddmaIndices, outputDims, ubSplitAxis,
                                            shapeLen - NDDMA_MAX_DIMS - 1 - ubSplitAxis);
        }
        int64_t nddmaUbOffset = BroadcastGetNddmaOffset(nddmaIndices, outputStrides, ubSplitAxis,
                                                        shapeLen - NDDMA_MAX_DIMS);
        int64_t nddmaGmOffset = BroadcastGetNddmaOffset(nddmaIndices, inputStrides, ubSplitAxis,
                                                        shapeLen - NDDMA_MAX_DIMS);
        if constexpr (AscendC::IsSameType<T1, T2>::value) {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor[nddmaUbOffset],
                                                          inputGm[gmOffset + nddmaGmOffset], paramsMain);
        } else {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor[nddmaUbOffset].template ReinterpretCast<T1>(),
                                                          inputGm[gmOffset + nddmaGmOffset], paramsMain);
        }
    }
}

template <typename T1, typename T2, size_t N>
__aicore__ __attribute__((noinline)) void BroadcastNddmaNonContiguousWithLoopNoInline(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    BroadcastNddmaNonContiguousWithLoop(inputGm, outputTensor, outputDims, outputStrides, inputStrides, axesIndices,
                                        ubSplitAxis, shapeLen, ubSplitSize, ubFormer);
}
} // namespace Base
} // namespace Ops
#endif // BROADCAST_UTILS_TRANSPOSE_H_