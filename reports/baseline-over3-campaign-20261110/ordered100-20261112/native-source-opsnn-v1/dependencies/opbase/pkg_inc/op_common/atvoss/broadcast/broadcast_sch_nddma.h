/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_NDDMA_SCH_H_
#define BROADCAST_NDDMA_SCH_H_

#include "broadcast_sch_base.h"

#pragma "lib"
namespace Ops {
namespace Base {
static constexpr int64_t NDDMA_COPYINBRC_INLINE_MAX_SUPPORT = 4;
static constexpr int64_t NDDMA_COPYINBRC_LEAST_NUMBER_NEED_FUSE = 3;

template <class BrcDag, bool withLoop>
class BroadcastNddmaSch : public BroadcastBaseSch<BrcDag, true> {
public:
    __aicore__ inline explicit BroadcastNddmaSch(const BroadcastBaseTilingData<BrcDag>* baseTilingData)
        : BroadcastBaseSch<BrcDag, true>(baseTilingData), tilingData(baseTilingData)
    {}

    template <class... Args>
    __aicore__ inline void Init(TPipe* pipe, Args... args)
    {
        static_assert(BrcDag::InputSize + BrcDag::OutputSize == sizeof...(Args),
                      "BroadcastNddmaSch.Init args num should match DAG holders.");
        static_assert(BrcDag::CopyBrcSize != 0, "Broadcast NDDMA schedule CopyInBrc node number should at least one.");

        InitInputArgs<0>(args...); // 调用入参分析,input, output
        this->template SetScalar<0>(0);
        RUN_LOG("BufferNum: %d, Mte2Num: %d, Mte3Num: %d, BufLevel: %d", bufferNum, BrcDag::Mte2Num, BrcDag::Mte3Num,
                BrcDag::BufLevel);

        TBuf<TPosition::VECCALC> buf;
        this->blockEleNum_ = tilingData->elemNum;
        this->blockLen_ = this->blockEleNum_ * BrcDag::MaxDtypeBytes;
        pipe->InitBuffer(buf, this->blockEleNum_ * BrcDag::MaxDtypeBytes * bufferNum);
        this->tensorPool_ = buf.Get<uint8_t>();

        RUN_LOG("GraphInfo Begin...");
        RUN_LOG("GraphInfo BrcDag::VecBrcSize is %d ", BrcDag::VecBrcSize);
        RUN_LOG("GraphInfo BrcDag::CopyBrcSize is %d ", BrcDag::CopyBrcSize);

        RUN_LOG("GraphInfo BrcDag::Mte2Num is %d ", BrcDag::Mte2Num);
        RUN_LOG("GraphInfo BrcDag::Mte3Num is %d ", BrcDag::Mte3Num);
        RUN_LOG("GraphInfo BrcDag::MaxAliveNodeForNddmaCacheBrc is %d ", BrcDag::MaxAliveNodeForNddmaCacheBrc);
        RUN_LOG("GraphInfo BrcDag::TempCalcNodeForNddmaCacheBrc is %d ", BrcDag::TempCalcNodeForNddmaCacheBrc);
        RUN_LOG("GraphInfo BrcDag::bufferNum is %d ", bufferNum);
        RUN_LOG("GraphInfo End...");

        RUN_LOG("Tiling Data Begin...");
        RUN_LOG("TilingData blockFormer is %d ", tilingData->blockFormer);
        RUN_LOG("TilingData ubFormer is %d ", tilingData->ubFormer);
        RUN_LOG("TilingData ubOuter is %d ", tilingData->ubOuter);
        RUN_LOG("TilingData ubTail is %d ", tilingData->ubTail);
        RUN_LOG("TilingData blockTail is %d ", tilingData->blockTail);
        RUN_LOG("TilingData shapeLen is %d ", tilingData->shapeLen);
        RUN_LOG("TilingData ubSplitAxis is %d ", tilingData->ubSplitAxis);
        RUN_LOG("TilingData dimProductBeforeUbInner is %d ", tilingData->dimProductBeforeUbInner);
        RUN_LOG("TilingData elemNum is %d ", this->blockEleNum_);
        RUN_LOG("TilingData numBlocks is %d ", AscendC::GetBlockNum());

        RUN_LOG("Tiling Data End...");
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingData->blockTail :
                                                                                   tilingData->blockFormer;
        int64_t axesIndices[BROADCAST_MAX_DIMS] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingData->blockFormer * AscendC::GetBlockIdx(), tilingData->outputDims,
                                tilingData->ubSplitAxis, tilingData->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            copyInBrcCount = 0;
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingData->outputDims, tilingData->ubSplitAxis,
                                           tilingData->ubOuter);
            }

            int64_t ubSplitSize = axesIndices[tilingData->ubSplitAxis] == tilingData->ubOuter - 1 ?
                                      tilingData->ubTail :
                                      tilingData->ubFormer;
            // ub整循环处理的元素个数
            Run<0, false>(ubSplitSize, axesIndices, ubLoopIdx, ubLoopIdx & 1);
        }
    }

protected:
    // 初始化输出
    template <int start, class... Args>
    __aicore__ inline void InitOutputArgs(GM_ADDR y, Args... args)
    {
        this->outGm_[start] = y;
        if constexpr (start + 1 < BrcDag::OutputSize) {
            InitOutputArgs<start + 1>(args...);
        }
    }

    // 初始化输入
    template <int start, class... Args>
    __aicore__ inline void InitInputArgs(GM_ADDR x, Args... args)
    {
        this->inGm_[start] = x;
        if constexpr (start + 1 < BrcDag::InputSize) {
            InitInputArgs<start + 1>(args...);
        } else {
            InitOutputArgs<0>(args...);
        }
    }

    template <typename Op, int pos>
    __aicore__ inline void CopyInBrc(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                     int64_t ubLoopIdx, int32_t pingPong)
    {
        static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");
        using input = typename Op::InHolders::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;

        int32_t bufId = this->template GetBufId<pos>(pingPong);
        RUN_LOG("CopyInBrc pingPong is %d ,pos is %d, bufId is %d", pingPong, pos, bufId);

        LocalTensor<inputType> inTensor = this->tensorPool_[bufId * this->blockLen_]
                                              .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inTensor.SetBufferLen(this->blockEleNum_);
#endif
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ inputType*>(this->inGm_[input::Pos]));

        RUN_LOG("copyInBrcCount is %d , input::Pos is %d ,pingpong is %d", copyInBrcCount, input::Pos, pingPong);
        if ((tilingData->inputBrcStrides[copyInBrcCount][tilingData->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter <= 1)) {
            GetTensor<TPosition::VECIN>(bufId);

            if constexpr (withLoop) {
                if constexpr (BrcDag::CopyBrcSize <= NDDMA_COPYINBRC_INLINE_MAX_SUPPORT) {
                    if constexpr (BrcDag::CopyBrcSize >= NDDMA_COPYINBRC_LEAST_NUMBER_NEED_FUSE) {
                        BroadcastNddmaWithLoopContiguousFuseAxis(
                            globalTensor, inTensor, tilingData->outputDims, tilingData->outputStrides,
                            tilingData->inputBrcStrides[copyInBrcCount], axesIndices, tilingData->ubSplitAxis,
                            tilingData->shapeLen, ubSplitSize, tilingData->ubFormer);
                    } else {
                        BroadcastNddmaWithLoop(globalTensor, inTensor, tilingData->outputDims,
                                               tilingData->outputStrides, tilingData->inputBrcStrides[copyInBrcCount],
                                               axesIndices, tilingData->ubSplitAxis, tilingData->shapeLen, ubSplitSize,
                                               tilingData->ubFormer);
                    }
                } else {
                    BroadcastNddmaWithLoopNoInline(
                        globalTensor, inTensor, tilingData->outputDims, tilingData->outputStrides,
                        tilingData->inputBrcStrides[copyInBrcCount], axesIndices, tilingData->ubSplitAxis,
                        tilingData->shapeLen, ubSplitSize, tilingData->ubFormer);
                }
            } else {
                if constexpr (BrcDag::CopyBrcSize <= NDDMA_COPYINBRC_INLINE_MAX_SUPPORT) {
                    BroadcastNddmaWithoutLoop(globalTensor, inTensor, tilingData->outputDims, tilingData->outputStrides,
                                              tilingData->inputBrcStrides[copyInBrcCount], axesIndices,
                                              tilingData->ubSplitAxis, tilingData->shapeLen, ubSplitSize,
                                              tilingData->ubFormer);
                } else {
                    BroadcastNddmaWithoutLoopNoInline(
                        globalTensor, inTensor, tilingData->outputDims, tilingData->outputStrides,
                        tilingData->inputBrcStrides[copyInBrcCount], axesIndices, tilingData->ubSplitAxis,
                        tilingData->shapeLen, ubSplitSize, tilingData->ubFormer);
                }
            }

            ReleaseTensor<TPosition::VECIN>(bufId);
        }

        copyInBrcCount = copyInBrcCount + 1;
    }

    // 遍历执行图
    template <int pos = 0, bool insideIf = false>
    __aicore__ inline void Run(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx,
                               int32_t pingPong)
    {
        if constexpr (pos >= BrcDag::FunList::Size) {
            return;
        }

        // Run current func
        using Op = typename BrcDag::FunList::template At<pos>;
        using Func = typename Op::Fun;
        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value && !insideIf) {
            constexpr int vecIndex = BrcDag::template VecBrcIdxDepend<Op>;
            if constexpr (vecIndex >= 0) {
                if ((tilingData->inputVecBrcStrides[vecIndex] != 0) ||
                    (ubLoopIdx == 0 ||
                     (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter == 0)) {
                    Run<pos, true>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                }
                using vecBrcOp = typename BrcDag::VecBrcNodes::template At<vecIndex>;
                constexpr int vecBrcOpPos = BrcDag::FunList::template GetIndex<vecBrcOp>();
                Run<vecBrcOpPos + 1, false>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                return;
            }
        }

        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
            this->template CopyIn<Op, pos>(axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyInBrc>::Value) {
            CopyInBrc<Op, pos>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
            int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->outputStrides, tilingData->ubSplitAxis,
                                                    tilingData->ubFormer);
            int64_t tileLength = ubSplitSize * tilingData->outputStrides[tilingData->ubSplitAxis];
            this->template CopyOut<Op, pos>(gmOffset, tileLength, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            this->template VecBroadcast<Op, pos>(ubLoopIdx, pingPong);
        } else {
            uint64_t tileLength = ubSplitSize * tilingData->outputStrides[tilingData->ubSplitAxis];
            this->template RunNormalOp<Op, pos>(tileLength, pingPong);
        }

        if constexpr (insideIf && __aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            return;
        }

        // Run next func
        if constexpr (pos + 1 < BrcDag::FunList::Size) {
            Run<pos + 1, insideIf>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        }
    }

private:
    constexpr static int bufferNum = BrcDag::template GetBufferNum<true, true>();
    int copyInBrcCount = 0;
    const BroadcastBaseTilingData<BrcDag>* tilingData;
};
} // namespace Base
} // namespace Ops
#endif // __BRC_NDDMA_DAG_H_
