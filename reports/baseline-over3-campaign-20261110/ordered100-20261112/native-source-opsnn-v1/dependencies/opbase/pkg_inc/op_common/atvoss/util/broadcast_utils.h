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
 * \file broadcast_utils.h
 * \brief
 */
#ifndef ASCENDC_BROADCAST_UTILS_H_
#define ASCENDC_BROADCAST_UTILS_H_

#include "atvoss/util/kernel_vec_intf_apt.h"
namespace Ops {
namespace Base {
static constexpr int64_t BROADCAST_MAX_DIMS = 8;
static constexpr int64_t NDDMA_MAX_DIMS = 5;
static constexpr int64_t NDDMA_THROW_DIMS = 3;

template <std::size_t N>
__aicore__ inline void BroadcastGetAxesIndices(int64_t (&axesIndices)[N], int64_t prodIdx,
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
__aicore__ inline void BroadcastUpdateAxesIndices(int64_t (&axesIndices)[N], const int64_t (&totalDims)[N],
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

template <std::size_t N, std::size_t M>
__aicore__ inline void BroadcastUpdateNddmaAxesIndices(int64_t (&axesIndices)[N], const int64_t (&totalDims)[M],
                                                       int64_t startIdx, int64_t endIdx)
{
    axesIndices[endIdx]++;
    for (int64_t idx = endIdx; idx >= 0; idx--) {
        if (axesIndices[idx] == totalDims[idx + startIdx]) {
            axesIndices[idx] = 0;
            axesIndices[idx - 1]++;
        }
    }
}

template <std::size_t N>
__aicore__ inline int64_t BroadcastGetGmOffset(const int64_t (&axesIndices)[N], const int64_t (&totalStrides)[N],
                                               int64_t endIdx, int64_t endDim)
{
    int64_t gmOffset = 0;
    for (int64_t idx = 0; idx < endIdx; idx++) {
        gmOffset += axesIndices[idx] * totalStrides[idx];
    }
    gmOffset += axesIndices[endIdx] * totalStrides[endIdx] * endDim;
    return gmOffset;
}

template <std::size_t N, std::size_t M>
__aicore__ inline int64_t BroadcastGetNddmaOffset(int64_t (&nddmaIndices)[N], const int64_t (&totalStrides)[M],
                                                  int64_t startIdx, int64_t endIdx)
{
    int64_t offset = 0;
    for (int64_t idx = 0; idx < endIdx - startIdx; idx++) {
        offset += nddmaIndices[idx] * totalStrides[idx + startIdx];
    }
    return offset;
}

template <std::size_t N>
__aicore__ inline int64_t BroadcastFuseAxes(const int64_t (&totalDims)[N], int64_t startIdx, int64_t endIdx)
{
    int64_t product = 1;
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        product *= totalDims[idx];
    }
    return product;
}

template <typename T, std::size_t N>
__aicore__ inline AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> BroadcastSetNddmaConfigWithoutLoop(
    const int64_t (&outputDims)[N], const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N],
    int64_t shapeLen, int64_t ubSplitSize, int64_t ubSplitAxis)
{
    AscendC::NdDmaLoopInfo<NDDMA_MAX_DIMS> loopInfo;
    int64_t axisInsideUb = NDDMA_MAX_DIMS - (shapeLen - ubSplitAxis);
    for (int64_t i = 0; i < axisInsideUb; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - i] = 1;
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - i] = inputStrides[ubSplitAxis];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - i] = outputStrides[ubSplitAxis];
    }

    loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - axisInsideUb] = ubSplitSize;
    loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - axisInsideUb] = inputStrides[ubSplitAxis];
    loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - axisInsideUb] = outputStrides[ubSplitAxis];

    for (uint64_t i = axisInsideUb + 1; i < NDDMA_MAX_DIMS; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - i] = outputDims[ubSplitAxis + i - axisInsideUb];
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - i] = inputStrides[ubSplitAxis + i - axisInsideUb];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - i] = outputStrides[ubSplitAxis + i - axisInsideUb];
    }

    T constValue = 0;
    AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> paramsMain = {loopInfo, constValue};
    return paramsMain;
}

template <typename T, std::size_t N>
__aicore__ inline AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> BroadcastSetNddmaConfigWithLoop(
    const int64_t (&outputDims)[N], const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N],
    int64_t shapeLen, int64_t ubSplitAxis)
{
    AscendC::NdDmaLoopInfo<NDDMA_MAX_DIMS> loopInfo;
    for (uint64_t i = 0; i < NDDMA_MAX_DIMS; i++) {
        loopInfo.loopSize[NDDMA_MAX_DIMS - 1 - i] = outputDims[i + shapeLen - NDDMA_MAX_DIMS];
        loopInfo.loopSrcStride[NDDMA_MAX_DIMS - 1 - i] = inputStrides[i + shapeLen - NDDMA_MAX_DIMS];
        loopInfo.loopDstStride[NDDMA_MAX_DIMS - 1 - i] = outputStrides[i + shapeLen - NDDMA_MAX_DIMS];
    }

    T constValue = 0;
    AscendC::NdDmaParams<T, NDDMA_MAX_DIMS> paramsMain = {loopInfo, constValue};
    return paramsMain;
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void DataCopyMoveAlign(AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor,
                                         const int64_t (&inputStrides)[N], int64_t ubSplitAxis, int64_t ubSplitSize,
                                         int64_t gmOffset)
{
    AscendC::DataCopyExtParams dataCopyExtParams;
    AscendC::DataCopyPadExtParams<T1> dataCopyPadExtParams;
    dataCopyExtParams.blockCount = 1;
    dataCopyExtParams.blockLen = ubSplitSize * inputStrides[ubSplitAxis] * sizeof(T1);
    if constexpr (AscendC::IsSameType<T1, T2>::value) {
        AscendC::DataCopyPad(outputTensor, inputGm[gmOffset], dataCopyExtParams, dataCopyPadExtParams);
    } else {
        AscendC::DataCopyPad(outputTensor.template ReinterpretCast<T1>(), inputGm[gmOffset], dataCopyExtParams,
                             dataCopyPadExtParams);
    }
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNddmaWithoutLoop(AscendC::GlobalTensor<T1>& inputGm,
                                                 AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
                                                 const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N],
                                                 const int64_t (&axesIndices)[N], int64_t ubSplitAxis, int64_t shapeLen,
                                                 int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {
        static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
        AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNddmaConfigWithoutLoop<T1>(
            outputDims, outputStrides, inputStrides, shapeLen, ubSplitSize, ubSplitAxis);
        if constexpr (AscendC::IsSameType<T1, T2>::value) {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor, inputGm[gmOffset], paramsMain);
        } else {
            AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor.template ReinterpretCast<T1>(),
                                                          inputGm[gmOffset], paramsMain);
        }
    } else {
        DataCopyMoveAlign(inputGm, outputTensor, inputStrides, ubSplitAxis, ubSplitSize, gmOffset);
    }
}

template <typename T1, typename T2, size_t N>
__aicore__ __attribute__((noinline)) void BroadcastNddmaWithoutLoopNoInline(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    BroadcastNddmaWithoutLoop(inputGm, outputTensor, outputDims, outputStrides, inputStrides, axesIndices, ubSplitAxis,
                              shapeLen, ubSplitSize, ubFormer);
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNddmaWithLoop(AscendC::GlobalTensor<T1>& inputGm,
                                              AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
                                              const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N],
                                              const int64_t (&axesIndices)[N], int64_t ubSplitAxis, int64_t shapeLen,
                                              int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {
        static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
        AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNddmaConfigWithLoop<T1>(
            outputDims, outputStrides, inputStrides, shapeLen, ubSplitAxis);
        int64_t nddmaIndices[NDDMA_THROW_DIMS] = {0};
        int64_t nddmaProduct = BroadcastFuseAxes(outputDims, ubSplitAxis + 1, shapeLen - NDDMA_MAX_DIMS) * ubSplitSize;
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
                AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(
                    outputTensor[nddmaUbOffset].template ReinterpretCast<T1>(), inputGm[gmOffset + nddmaGmOffset],
                    paramsMain);
            }
        }
    } else {
        DataCopyMoveAlign(inputGm, outputTensor, inputStrides, ubSplitAxis, ubSplitSize, gmOffset);
    }
}

template <typename T1, typename T2, size_t N>
__aicore__ __attribute__((noinline)) void BroadcastNddmaWithLoopNoInline(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    BroadcastNddmaWithLoop(inputGm, outputTensor, outputDims, outputStrides, inputStrides, axesIndices, ubSplitAxis,
                           shapeLen, ubSplitSize, ubFormer);
}

template <typename T1, typename T2, size_t N>
__aicore__ inline void BroadcastNddmaWithLoopContiguousFuseAxis(
    AscendC::GlobalTensor<T1>& inputGm, AscendC::LocalTensor<T2>& outputTensor, const int64_t (&outputDims)[N],
    const int64_t (&outputStrides)[N], const int64_t (&inputStrides)[N], const int64_t (&axesIndices)[N],
    int64_t ubSplitAxis, int64_t shapeLen, int64_t ubSplitSize, int64_t ubFormer)
{
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);
    if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {
        int64_t outputDims2[BROADCAST_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
        int64_t outputStrides2[BROADCAST_MAX_DIMS] = {0};
        int64_t inputStrides2[BROADCAST_MAX_DIMS] = {0};
        int64_t count = shapeLen - 1;
        int64_t curFlag = 0;
        int64_t oriFlag = -1;
        int64_t newCount = shapeLen;

        while (count > ubSplitAxis) {
            curFlag = inputStrides[count] == 0 ? 0 : 1;
            if (curFlag != oriFlag) {
                newCount = newCount - 1;
                outputDims2[newCount] = outputDims[count];
                outputStrides2[newCount] = outputStrides[count];
                inputStrides2[newCount] = inputStrides[count];
                oriFlag = curFlag;
            } else {
                outputDims2[newCount] = outputDims2[newCount] * outputDims[count];
            }
            count--;
        }

        outputDims2[newCount - 1] = ubSplitSize;
        outputStrides2[newCount - 1] = outputStrides[count];
        inputStrides2[newCount - 1] = inputStrides[count];

        static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
        AscendC::NdDmaParams<T1, NDDMA_MAX_DIMS> paramsMain = BroadcastSetNddmaConfigWithLoop<T1>(
            outputDims2, outputStrides2, inputStrides2, shapeLen, ubSplitAxis);
        int64_t nddmaIndices[NDDMA_THROW_DIMS] = {0};
        int64_t nddmaProduct = BroadcastFuseAxes(outputDims2, ubSplitAxis, shapeLen - NDDMA_MAX_DIMS);
        for (int64_t i = 0; i < nddmaProduct; i++) {
            if (i != 0) {
                BroadcastUpdateNddmaAxesIndices(nddmaIndices, outputDims2, ubSplitAxis,
                                                shapeLen - NDDMA_MAX_DIMS - 1 - ubSplitAxis);
            }
            int64_t nddmaGmOffset = BroadcastGetNddmaOffset(nddmaIndices, inputStrides2, ubSplitAxis,
                                                            shapeLen - NDDMA_MAX_DIMS);
            int64_t nddmaUbOffset = BroadcastGetNddmaOffset(nddmaIndices, outputStrides2, ubSplitAxis,
                                                            shapeLen - NDDMA_MAX_DIMS);
            if constexpr (AscendC::IsSameType<T1, T2>::value) {
                AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(outputTensor[nddmaUbOffset],
                                                              inputGm[gmOffset + nddmaGmOffset], paramsMain);
            } else {
                AscendC::DataCopy<T1, NDDMA_MAX_DIMS, config>(
                    outputTensor[nddmaUbOffset].template ReinterpretCast<T1>(), inputGm[gmOffset + nddmaGmOffset],
                    paramsMain);
            }
        }
    } else {
        DataCopyMoveAlign(inputGm, outputTensor, inputStrides, ubSplitAxis, ubSplitSize, gmOffset);
    }
}

} // namespace Base
} // namespace Ops
#endif