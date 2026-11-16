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
 * \file broadcast_sch_last_transpose.h
 * \brief
 */
#ifndef BROADCAST_LAST_TRANSPOSE_SCH_H_
#define BROADCAST_LAST_TRANSPOSE_SCH_H_

#include "broadcast_sch_base_last_transpose.h"
#include "broadcast_sch_util_last_transpose.h"

#pragma "lib"
namespace Ops {
namespace Base {
template <class BrcDag>
class BroadcastLastTransposeSch : public BroadcastBaseSchLastTranspose<BrcDag, true> {
public:
    __aicore__ inline explicit BroadcastLastTransposeSch(const BroadcastLastTransposeTilingData<BrcDag>* baseTilingData)
        : BroadcastBaseSchLastTranspose<BrcDag, true>(baseTilingData), tilingData(baseTilingData)
    {}

    template <class... Args>
    __aicore__ inline void Init(TPipe* pipe, Args... args)
    {
        static_assert(BrcDag::InputSize + BrcDag::OutputSize == sizeof...(Args),
                      "BroadcastLastTransposeSch.Init args num should match DAG holders.");
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
        int64_t axesIndices[BROADCAST_MAX_DIMS_LAST_TRANSPOSE] = {0};

        // 获取多核偏移信息
        BroadcastGetAxesIndices(axesIndices, tilingData->blockFormer * AscendC::GetBlockIdx(), tilingData->outputDims,
                                tilingData->ubSplitAxis, tilingData->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                // 更新UB间循环偏移信息
                BroadcastUpdateAxesIndices(axesIndices, tilingData->outputDims, tilingData->ubSplitAxis,
                                           tilingData->ubOuter);
            }

            int64_t notLastUBSplitSize = axesIndices[tilingData->ubSplitAxis] == tilingData->ubOuter - 1 ?
                                             tilingData->ubTail :
                                             tilingData->ubFormer;

            // ub整循环处理的元素个数
            for (int64_t lastAxisLoopIdx = 0; lastAxisLoopIdx < tilingData->ubOuterLastAxis; lastAxisLoopIdx += 1) {
                copyInBrcCount = 0;
                int64_t lastUBSplitSize = lastAxisLoopIdx == tilingData->ubOuterLastAxis - 1 ?
                                              tilingData->ubTailLastAxis :
                                              tilingData->ubFormerLastAxis;
                Run<0, false>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize, lastAxisLoopIdx & 1);
            }
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
    __aicore__ inline void CopyInBrc(const int64_t (&axesIndices)[BROADCAST_MAX_DIMS_LAST_TRANSPOSE],
                                     int64_t lastAxisLoopIdx, int64_t notLastUBSplitSize, int64_t lastUBSplitSize,
                                     int32_t pingPong)
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
        int64_t gmOffset = BroadcastGetGmOffsetLastTranspose(
            axesIndices, tilingData->inputBrcStrides[copyInBrcCount], tilingData->ubSplitAxis, tilingData->ubFormer,
            tilingData->shapeLen, lastAxisLoopIdx, tilingData->ubFormerLastAxis);
        GlobalTensor<inputType> inputGm;
        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(this->inGm_[input::Pos] + gmOffset * sizeof(inputType)));

        RUN_LOG("copyInBrcCount is %d , input::Pos is %d ,pingpong is %d", copyInBrcCount, input::Pos, pingPong);

        // 当内循环是last轴时，看last轴是不是brc轴. 由于此时last轴不参与合轴，所以只要满足前两次不缓存即可。
        // 当内循环是ub切分轴时，看ub切分轴tilingData->ubSplitAxis是不是brc轴.
        if (tilingData->inputBrcStrides[copyInBrcCount][tilingData->shapeLen - 1] != 0 || lastAxisLoopIdx <= 1 ||
            lastAxisLoopIdx == tilingData->ubOuterLastAxis - 1) {
            GetTensor<TPosition::VECIN>(bufId);

            // 只支持最多3维，所以直接使用不需要循环的nddma即可。
            int64_t inputUBSize = BroadcastFuseAxes(tilingData->inputBrcDims[copyInBrcCount],
                                                    tilingData->ubSplitAxis + 1, tilingData->shapeLen);

            // 走NDDMA的条件
            // 1. UB 切分轴为brc轴
            // 2. UB切分轴外的UB内轴乘积输入不等于输出
            // 3. last轴转置
            if (tilingData->inputBrcStrides[copyInBrcCount][tilingData->ubSplitAxis] == 0 ||
                tilingData->outputStrides[tilingData->ubSplitAxis] != inputUBSize ||
                tilingData->inputBrcStrides[copyInBrcCount][tilingData->shapeLen - 1] != 1) {
                static constexpr AscendC::NdDmaConfig config = {false, 0, 0, false};
                AscendC::NdDmaParams<inputType, NDDMA_MAX_DIMS_LAST_TRANSPOSE>
                    paramsMain = BroadcastSetNddmaConfigWithoutLoop<inputType>(
                        tilingData->outputDims, tilingData->outputStridesWithPad,
                        tilingData->inputBrcStrides[copyInBrcCount], tilingData->shapeLen, tilingData->ubSplitAxis,
                        notLastUBSplitSize, lastUBSplitSize);
                AscendC::DataCopy<inputType, NDDMA_MAX_DIMS_LAST_TRANSPOSE, config>(inTensor, inputGm, paramsMain);
            } else {
                // Run copyIn
                MovAlign2UB<inputType>(inTensor, inputGm, tilingData->outputDims,
                                       tilingData->inputBrcStrides[copyInBrcCount], tilingData->outputStridesWithPad,
                                       tilingData->ubSplitAxis, tilingData->shapeLen, notLastUBSplitSize,
                                       lastUBSplitSize);
            }

            ReleaseTensor<TPosition::VECIN>(bufId);
        }

        copyInBrcCount = copyInBrcCount + 1;
    }

    // 遍历执行图
    template <int pos = 0, bool insideIf = false>
    __aicore__ inline void Run(const int64_t (&axesIndices)[BROADCAST_MAX_DIMS_LAST_TRANSPOSE], int64_t lastAxisLoopIdx,
                               int64_t notLastUBSplitSize, int64_t lastUBSplitSize, int32_t pingPong)
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
                if ((tilingData->inputVecBrcStrides[vecIndex] != 0) || lastAxisLoopIdx == 0) {
                    Run<pos, true>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize, pingPong);
                }
                using vecBrcOp = typename BrcDag::VecBrcNodes::template At<vecIndex>;
                constexpr int vecBrcOpPos = BrcDag::FunList::template GetIndex<vecBrcOp>();
                Run<vecBrcOpPos + 1, false>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize,
                                            pingPong);
                return;
            }
        }

        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
            this->template CopyIn<Op, pos>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyInBrc>::Value) {
            CopyInBrc<Op, pos>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
            this->template CopyOut<Op, pos>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize,
                                            pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            this->template VecBroadcast<Op, pos>(lastAxisLoopIdx, pingPong);
        } else {
            this->template RunNormalOp<Op, pos>(tilingData->elemNum, pingPong);
        }

        if constexpr (insideIf && __aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            return;
        }

        // Run next func
        if constexpr (pos + 1 < BrcDag::FunList::Size) {
            Run<pos + 1, insideIf>(axesIndices, lastAxisLoopIdx, notLastUBSplitSize, lastUBSplitSize, pingPong);
        }
    }

private:
    constexpr static int bufferNum = BrcDag::template GetBufferNum<true, true>();
    int copyInBrcCount = 0;
    const BroadcastLastTransposeTilingData<BrcDag>* tilingData;
};
} // namespace Base
} // namespace Ops
#endif // BROADCAST_LAST_TRANSPOSE_SCH_H_
