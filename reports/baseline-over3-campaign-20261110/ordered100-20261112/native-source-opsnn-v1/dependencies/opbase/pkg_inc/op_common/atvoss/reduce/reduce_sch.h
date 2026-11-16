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
 * \file reduce_sch.h
 * \brief reduce schedule
 */

#ifndef _REDUCE_SCH_H_
#define _REDUCE_SCH_H_
#include "reduce_tiling_data.h"
#include "reduce_sch_aux.h"
#include "reduce_sch_aux_non_contiguous.h"
#include "reduce_sch_aux_util.h"
#include "reduce_operator.h"
#include "reduce_tensor_empty.h"
#include "reduce_tensor_move.h"
#include "reduce_sch_batch_invariant.h"

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {
template <class ElemOp, bool IsVoid>
struct EleSclar {
    using T = typename ElemOp::VarType;
};

template <class ElemOp>
struct EleSclar<ElemOp, true> {
    using T = void;
};

template <bool isContiguous, bool batchInvariant, uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount,
          class OpDag>
class ReduceSch {
public:
    constexpr static ReduceSchLoopInfo SchLoopInfo = GetSchLoopInfo<PatternID, LoopARCount, LoopInnerARCount>();
    using Pattern = typename __reducePattern::GetPattern<SchLoopInfo.patternID>::T;
    using ReduceOpBind = typename OpDag::FunList::template At<OpDag::ReduceOpPos>;
    using ReduceOp = typename ReduceOpBind::Fun;
    using DataType = typename ReduceOpBind::template FunInArgType<0>;
    using InDType = typename InputDType<OpDag, DataType, true, IsSameType<OpDag, void>::value>::T;
    using OutDType = typename OutputDType<OpDag, DataType, OpDag::FunList::Size - 1, IsSameType<OpDag, void>::value>::T;
    constexpr static int32_t ELEMENT_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(DataType);
    constexpr static uint64_t UB_BLOCK = Ops::Base::GetUbBlockSize();
    constexpr static int32_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static bool isBatchInvariant = batchInvariant;
    ReduceOp* reduceOp_;

private:
    // 无DAG的场景, 默认单算子为单输入单输出
    __aicore__ constexpr static int32_t GetPreMte2Num()
    {
        if constexpr (OpDag::ReduceOpPos == 1) {
            return OpDag::template GetMte2NumImpl<typename OpDag::PreReduceNodeInfo, true, false>() * CONST2 * CONST2;
        } else {
            return OpDag::template GetMte2NumImpl<typename OpDag::PreReduceNodeInfo, true, false>() * CONST2;
        }
    }

    __aicore__ constexpr static int32_t GetPreMte3Num()
    {
        return OpDag::template GetMte3NumImpl<typename OpDag::PreReduceNodeInfo, true, false>() * CONST2;
    }

    __aicore__ constexpr static int32_t GetPreTempCalcNum()
    {
        return OpDag::template GetTempBufNumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
    }

    __aicore__ constexpr static int32_t GetPostMte2Num()
    {
        return OpDag::template GetMte2NumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
    }

    __aicore__ constexpr static int32_t GetPostMte3Num()
    {
        return OpDag::template GetMte3NumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
    }

    __aicore__ constexpr static int32_t GetPostTempCalcNum()
    {
        return OpDag::template GetTempBufNumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
    }

    __aicore__ inline constexpr static int32_t Ratio()
    {
        if constexpr (OpDag::BufLevel == MemLevel::LEVEL_2) {
            return OpDag::MaxDtypeBytes / sizeof(InDType);
        } else {
            return 1;
        }
    }

    __aicore__ inline constexpr static bool IsEmpty()
    {
        return PatternID == 0 && LoopARCount == 0 && LoopInnerARCount == 0;
    }

    __aicore__ inline constexpr static bool IsTensorMove()
    {
        // if pre reduce only have copyin
        if constexpr (Pattern::ID == PATTERN_A &&
                      (OpDag::ReduceOpPos == 1 && OpDag::PostReduceNodeInfo::SavedFunList::Size == 1)) {
            return true;
        } else {
            return false;
        }
    }

private:
    const ReduceOpTilingData* tiling_;
    TPipe* pipeIn_;
    constexpr static int32_t PRE_MTE2_NUM_TOTAL = GetPreMte2Num();
    constexpr static int32_t PRE_MTE3_NUM_TOTAL = GetPreMte3Num();
    constexpr static int32_t PRE_TEMP_CALC_NUM = GetPreTempCalcNum();
    constexpr static int32_t POST_MTE2_NUM = GetPostMte2Num();
    constexpr static int32_t POST_MTE3_NUM = GetPostMte3Num();
    constexpr static int32_t POST_TEMP_CALC_NUM = GetPostTempCalcNum();
    constexpr static int32_t
        PRE_MTE2_NUM = OpDag::template GetMte2NumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
    constexpr static int32_t
        PRE_MTE3_NUM = OpDag::template GetMte3NumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
    constexpr static uint32_t INPUT_NUM = OpDag::InputSize;
    constexpr static uint32_t OUPUT_NUM = OpDag::OutputSize;
    constexpr static MemLevel BUF_LEVEL = OpDag::BufLevel;
    typename EleSclar<OpDag, IsSameType<OpDag, void>::value>::T OpScalars_;

    GlobalTensor<uint8_t> input_[INPUT_NUM];
    GlobalTensor<uint8_t> output_[OUPUT_NUM];
    GlobalTensor<uint8_t> workspace_;
    LocalTensor<uint8_t> preBufPool_;
    LocalTensor<uint8_t> postBufPool_;
    LocalTensor<uint8_t> resBuf_;
    LocalTensor<uint8_t> cache_;
    TBuf<TPosition::VECCALC> buf_;
    int32_t preBufSize_ = 0;
    int32_t postBufSize_ = 0;
    int32_t resBufSize_ = 0;

protected:
    template <class... Args>
    __aicore__ inline void InitArgsWorkspace(GM_ADDR workspace, Args... args)
    {
        workspace_.SetGlobalBuffer((__gm__ uint8_t*)workspace);
    }

    template <int start, class... Args>
    __aicore__ inline void InitArgsOutput(GM_ADDR y, Args... args)
    {
        output_[start].SetGlobalBuffer((__gm__ uint8_t*)y);
        if constexpr (start + 1 < OUPUT_NUM) {
            InitArgsOutput<start + 1>(args...);
        } else {
            InitArgsWorkspace(args...);
        }
    }

    template <int start, class... Args>
    __aicore__ inline void InitArgsInput(GM_ADDR x, Args... args)
    {
        input_[start].SetGlobalBuffer((__gm__ uint8_t*)x);
        if constexpr (start + 1 < INPUT_NUM) {
            InitArgsInput<start + 1>(args...);
        } else {
            InitArgsOutput<0>(args...);
        }
    }

public:
    __aicore__ inline explicit ReduceSch(const ReduceOpTilingData* tiling) { tiling_ = tiling; };

    template <class... Args>
    __aicore__ inline void Init(TPipe* pipeIn, Args... args)
    {
        static_assert(BUF_LEVEL != MemLevel::LEVEL_0, "Buffer Level should be in [1, 2,]");
        pipeIn_ = pipeIn;
        InitArgsInput<0>(args...);
        if constexpr (CheckInit<ReduceOp>(0)) {
            reduceOp_->Init(args...);
        }

        if constexpr (IsEmpty() || IsTensorMove()) {
            // empty tensor and pure move, init buf in its own template
            return;
        }

        preBufSize_ = (tiling_->basicBlock * PRE_MTE2_NUM_TOTAL) +
                      (tiling_->basicBlock * Ratio() * (PRE_MTE3_NUM_TOTAL + PRE_TEMP_CALC_NUM));
        postBufSize_ = tiling_->resultBlock * (POST_MTE2_NUM + POST_TEMP_CALC_NUM + POST_MTE3_NUM);
        resBufSize_ = tiling_->resultBlock;

        RUN_LOG("Pre MTE2: %d, Calc:%d, MTE3:%d, Post MTE2:%d, Calc:%d, MTE3:%d, preBufSize:%d, postBufSize:%d, "
                "resBufSize:%d\n",
                PRE_MTE2_NUM_TOTAL, PRE_TEMP_CALC_NUM, PRE_MTE3_NUM_TOTAL, POST_MTE2_NUM, POST_TEMP_CALC_NUM,
                POST_MTE3_NUM, preBufSize_, postBufSize_, resBufSize_);

        pipeIn->InitBuffer(buf_, preBufSize_ + postBufSize_ + resBufSize_ + CACHE_BUF_SIZE);
        preBufPool_ = buf_.Get<uint8_t>();
        postBufPool_ = preBufPool_[preBufSize_];
        resBuf_ = preBufPool_[preBufSize_ + postBufSize_];
        cache_ = preBufPool_[preBufSize_ + postBufSize_ + resBufSize_];
    }

    template <class... Args>
    __aicore__ inline void ProcessEmpty(Args... args)
    {
        using OpEmpty = ReduceTensorEmpty<OutDType>;
        OpEmpty op(tiling_, output_, pipeIn_, args...);
        op.Process();
    }

    template <class... Args>
    __aicore__ inline void ProcessMove(Args... args)
    {
        using OpMove = ReduceTensorMove<InDType>;
        OpMove op(tiling_, input_, output_, pipeIn_);
        op.Process();
    }

    template <class... Args>
    __aicore__ inline void ProcessNormal(Args... args)
    {
        if constexpr (isContiguous) {
            using SchTypeA = ReduceSchAux<&SchLoopInfo, std::remove_reference_t<decltype(*this)>, true, OpDag>;
            SchTypeA op(this, input_, output_, &workspace_, tiling_);
            op.Process(args...);
        } else {
            using SchTypeA = ReduceSchAuxNonContiguous<&SchLoopInfo, std::remove_reference_t<decltype(*this)>, true,
                                                       OpDag>;
            SchTypeA op(this, input_, output_, &workspace_, tiling_);
            op.Process(args...);
        }
    }

    template <class... Args>
    __aicore__ inline void ProcessGroup(Args... args)
    {
        constexpr static ReduceSchLoopInfo groupSchLoopInfo = GetGroupSchLoopInfo();
        ReduceOpTilingData groupTiling;
        SetGroupTiling(groupTiling);
        if constexpr (isContiguous && !batchInvariant) {
            using SchTypeR = ReduceSchAux<&SchLoopInfo, std::remove_reference_t<decltype(*this)>, true, OpDag>;
            SchTypeR op(this, input_, &workspace_, &workspace_, tiling_);
            using SchTypeA = ReduceSchAux<&groupSchLoopInfo, std::remove_reference_t<decltype(*this)>, false, OpDag>;
            SchTypeA groupOp(this, input_, output_, &workspace_, &groupTiling);

            op.Process(args...);
            SyncAll();
            groupOp.Process(args...);
        } else if constexpr (isContiguous && batchInvariant) {
            groupTiling.ubFactorA = tiling_->resultBlock / sizeof(DataType);
            groupTiling.factorACntPerCore = Ops::Base::CeilDiv(groupTiling.factorATotalCnt,
                                                               static_cast<uint64_t>(tiling_->realCoreNum));
            using SchTypeR = ReduceSchAuxBatchInvariant<&SchLoopInfo, std::remove_reference_t<decltype(*this)>, true,
                                                        OpDag>;
            SchTypeR op(this, input_, &workspace_, &workspace_, tiling_);
            using SchTypeA = ReduceSchAuxBatchInvariant<&groupSchLoopInfo, std::remove_reference_t<decltype(*this)>,
                                                        false, OpDag>;
            SchTypeA groupOp(this, input_, output_, &workspace_, &groupTiling);

            op.Process(args...);
            SyncAll();
            groupOp.Process(args...);
        } else {
            SetGroupTilingNonContiguous(groupTiling);
            using SchTypeR = ReduceSchAuxNonContiguous<&SchLoopInfo, std::remove_reference_t<decltype(*this)>, true,
                                                       OpDag>;
            SchTypeR op(this, input_, &workspace_, &workspace_, tiling_);
            using SchTypeA = ReduceSchAuxNonContiguous<&groupSchLoopInfo, std::remove_reference_t<decltype(*this)>,
                                                       false, OpDag>;
            SchTypeA groupOp(this, input_, output_, &workspace_, &groupTiling);

            op.Process(args...);
            SyncAll();
            groupOp.Process(args...);
        }
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        if constexpr (CheckProcess<ReduceOp>(0)) {
            reduceOp_->Process(args...);
        } else {
            if constexpr (IsEmpty()) {
                ProcessEmpty(args...);
            } else if constexpr (IsTensorMove()) {
                ProcessMove(args...);
            } else if constexpr (SchLoopInfo.loopRCount == 0) {
                ProcessNormal(args...);
            } else {
                ProcessGroup(args...);
            }
        }
    }

    __aicore__ inline void SetGroupTiling(ReduceOpTilingData& groupTiling)
    {
        groupTiling.ubFactorA = ELEMENT_ONE_REPEAT;
        groupTiling.ubFactorR = tiling_->groupR;
        groupTiling.shape[0] = tiling_->groupR;
        groupTiling.shape[1] = tiling_->outSize;
        groupTiling.stride[0] = Ops::Base::CeilAlign(tiling_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT));
        groupTiling.stride[1] = 1;
        groupTiling.dstStride[0] = tiling_->outSize;
        groupTiling.dstStride[1] = 1;
        groupTiling.groupR = 1;
        groupTiling.outSize = tiling_->outSize;
        groupTiling.basicBlock = tiling_->basicBlock;
        groupTiling.resultBlock = tiling_->resultBlock;
        groupTiling.useNddma = 0;
        groupTiling.factorRCntPerCore = 1;
        groupTiling.factorRTotalCnt = 1;
        groupTiling.factorATotalCnt = Ops::Base::CeilDiv(groupTiling.shape[1], groupTiling.ubFactorA);
        groupTiling.factorACntPerCore = Ops::Base::CeilDiv(groupTiling.factorATotalCnt,
                                                           static_cast<uint64_t>(tiling_->coreNum));
    }

    __aicore__ inline void SetGroupTilingNonContiguous(ReduceOpTilingData& groupTiling)
    {
        groupTiling.sliceNum[0] = 1;
        groupTiling.sliceNum[1] = 1;
        groupTiling.sliceShape[0] = tiling_->groupR;
        groupTiling.sliceShape[1] = tiling_->outSize;
        groupTiling.sliceStride[0] = Ops::Base::CeilAlign(tiling_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT));
        groupTiling.sliceStride[1] = 1;
    }

    template <class T, class V>
    __aicore__ inline void MaxLoopDataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view,
                                              const DataCopyExtParams& copyInParams,
                                              const DataCopyPadExtParams<T>& padParams)
    {
        for (uint64_t i = 0; i < view.axis[CONST7].repeat; i++) {
            for (uint64_t j = 0; j < view.axis[CONST6].repeat; j++) {
                for (uint64_t k = 0; k < view.axis[CONST5].repeat; k++) {
                    for (uint64_t l = 0; l < view.axis[CONST4].repeat; l++) {
                        int64_t dstStride = i * view.axis[CONST7].dstStride + j * view.axis[CONST6].dstStride +
                                            k * view.axis[CONST5].dstStride + l * view.axis[CONST4].dstStride;
                        int64_t srcStride = i * view.axis[CONST7].srcStride + j * view.axis[CONST6].srcStride +
                                            k * view.axis[CONST5].srcStride + l * view.axis[CONST4].srcStride;
                        DataCopyPad(dst[dstStride], src[view.addr + srcStride], copyInParams, padParams);
                    }
                }
            }
        }
    }

    template <class T, class V>
    __aicore__ inline void InitDataCopyPadParams(DataCopyExtParams& copyInParams, LoopModeParams& loopParams, V& view)
    {
        copyInParams.blockCount = view.axis[CONST1].repeat;
        copyInParams.blockLen = view.axis[CONST0].repeat * sizeof(T);
        copyInParams.srcStride = (view.axis[CONST1].srcStride - view.axis[CONST0].repeat) * sizeof(T);
        copyInParams.dstStride = (view.axis[CONST1].dstStride - view.axis[CONST0].repeat) * sizeof(T) / UB_BLOCK;

        loopParams.loop1Size = view.axis[CONST2].repeat;
        loopParams.loop1SrcStride = view.axis[CONST2].srcStride * sizeof(T);
        loopParams.loop1DstStride = view.axis[CONST2].dstStride * sizeof(T);
        loopParams.loop2Size = view.axis[CONST3].repeat;
        loopParams.loop2SrcStride = view.axis[CONST3].srcStride * sizeof(T);
        loopParams.loop2DstStride = view.axis[CONST3].dstStride * sizeof(T);
    }

    template <int32_t pos, class InnerPattern, class T, class V>
    __aicore__ inline void CopyInWithMoveAlign(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view)
    {
        T paddingValue = 0;
        DataCopyPadExtParams<T> padParams{true, 0, 0, paddingValue};
        DataCopyExtParams copyInParams;
        LoopModeParams loopParams;
        InitDataCopyPadParams<T>(copyInParams, loopParams, view);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        if constexpr (Pattern::Dim <= CONST4) {
            DataCopyPad(dst, src[view.addr], copyInParams, padParams);
        } else {
            MaxLoopDataCopyPad(dst, src, view, copyInParams, padParams);
        }
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    template <int32_t pos, class InnerPattern, class T, class V>
    __aicore__ inline void CopyInWithMoveAlignNonContiguous(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                                            V& view)
    {
        T paddingValue = 0;
        DataCopyPadExtParams<T> padParams{true, 0, 0, paddingValue};
        DataCopyExtParams copyInParams;
        LoopModeParams loopParams;
        InitDataCopyPadParams<T>(copyInParams, loopParams, view);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        MaxLoopDataCopyPad(dst, src, view, copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    template <int32_t pos, class InnerPattern, class T, class V>
    __aicore__ inline void CopyInWithNddma(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view)
    {
        uint64_t outer = 1;
        for (int32_t i = CONST2; i < view.axisSize; i += CONST1) {
            outer = outer * view.axis[i].repeat;
        }
        if (outer == 1) {
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = view.axis[CONST1].repeat;
            copyInParams.blockLen = view.axis[CONST0].repeat * sizeof(T);
            copyInParams.srcStride = (view.axis[CONST1].srcStride - view.axis[CONST0].repeat) * sizeof(T);
            copyInParams.dstStride = (view.axis[CONST1].dstStride - view.axis[CONST0].repeat) * sizeof(T) / UB_BLOCK;
            DataCopyPad(dst, src[view.addr], copyInParams, padParams);
        } else {
            static constexpr MultiCopyConfig config = {false, 0, 0, false};
            if constexpr (Pattern::Dim <= CONST4) {
                if constexpr (Pattern::Dim == CONST3) {
                    MultiCopyLoopInfo<CONST3> copyLoopInfo = {
                        .loopSrcStride = {1, view.axis[CONST1].srcStride, view.axis[CONST2].srcStride},
                        .loopDstStride = {1, static_cast<uint32_t>(view.axis[CONST1].dstStride),
                                          static_cast<uint32_t>(view.axis[CONST2].dstStride)},
                        .loopSize = {static_cast<uint32_t>(view.axis[CONST0].repeat),
                                     static_cast<uint32_t>(view.axis[CONST1].repeat),
                                     static_cast<uint32_t>(view.axis[CONST2].repeat)},
                        .loopLpSize = {0, 0, 0},
                        .loopRpSize = {0, 0, 0}};
                    MultiCopyParams<T, CONST3> params = {copyLoopInfo, 0};
                    DataCopy<T, CONST3, config>(dst, src[view.addr], params);
                } else {
                    MultiCopyLoopInfo<CONST4> copyLoopInfo = {
                        .loopSrcStride = {1, view.axis[CONST1].srcStride, view.axis[CONST2].srcStride,
                                          view.axis[CONST3].srcStride},
                        .loopDstStride = {1, static_cast<uint32_t>(view.axis[CONST1].dstStride),
                                          static_cast<uint32_t>(view.axis[CONST2].dstStride),
                                          static_cast<uint32_t>(view.axis[CONST3].dstStride)},
                        .loopSize = {static_cast<uint32_t>(view.axis[CONST0].repeat),
                                     static_cast<uint32_t>(view.axis[CONST1].repeat),
                                     static_cast<uint32_t>(view.axis[CONST2].repeat),
                                     static_cast<uint32_t>(view.axis[CONST3].repeat)},
                        .loopLpSize = {0, 0, 0, 0},
                        .loopRpSize = {0, 0, 0, 0}};
                    MultiCopyParams<T, CONST4> params = {copyLoopInfo, 0};
                    DataCopy<T, CONST4, config>(dst, src[view.addr], params);
                }
            } else {
                MultiCopyLoopInfo<CONST5> copyLoopInfo = {
                    .loopSrcStride = {1, view.axis[CONST1].srcStride, view.axis[CONST2].srcStride,
                                      view.axis[CONST3].srcStride, view.axis[CONST4].srcStride},
                    .loopDstStride = {1, static_cast<uint32_t>(view.axis[CONST1].dstStride),
                                      static_cast<uint32_t>(view.axis[CONST2].dstStride),
                                      static_cast<uint32_t>(view.axis[CONST3].dstStride),
                                      static_cast<uint32_t>(view.axis[CONST4].dstStride)},
                    .loopSize = {static_cast<uint32_t>(view.axis[CONST0].repeat),
                                 static_cast<uint32_t>(view.axis[CONST1].repeat),
                                 static_cast<uint32_t>(view.axis[CONST2].repeat),
                                 static_cast<uint32_t>(view.axis[CONST3].repeat),
                                 static_cast<uint32_t>(view.axis[CONST4].repeat)},
                    .loopLpSize = {0, 0, 0, 0, 0},
                    .loopRpSize = {0, 0, 0, 0, 0}};
                MultiCopyParams<T, CONST5> params = {copyLoopInfo, 0};
                for (uint64_t i = 0; i < view.axis[CONST7].repeat; i++) {
                    for (uint64_t j = 0; j < view.axis[CONST6].repeat; j++) {
                        for (uint64_t k = 0; k < view.axis[CONST5].repeat; k++) {
                            int64_t dstStride = i * view.axis[CONST7].dstStride + j * view.axis[CONST6].dstStride +
                                                k * view.axis[CONST5].dstStride;
                            int64_t srcStride = i * view.axis[CONST7].srcStride + j * view.axis[CONST6].srcStride +
                                                k * view.axis[CONST5].srcStride;
                            DataCopy<T, CONST5, config>(dst[dstStride], src[view.addr + srcStride], params);
                        }
                    }
                }
            }
        }
    }

    template <int32_t pos, class InnerPattern, class T, class V>
    __aicore__ inline void CopyInWithNddmaNonContiguous(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view)
    {
        static constexpr MultiCopyConfig config = {false, 0, 0, false};
        MultiCopyLoopInfo<CONST5> copyLoopInfo = {
            .loopSrcStride = {view.axis[CONST0].srcStride, view.axis[CONST1].srcStride, view.axis[CONST2].srcStride,
                              view.axis[CONST3].srcStride, view.axis[CONST4].srcStride},
            .loopDstStride = {static_cast<uint32_t>(view.axis[CONST0].dstStride),
                              static_cast<uint32_t>(view.axis[CONST1].dstStride),
                              static_cast<uint32_t>(view.axis[CONST2].dstStride),
                              static_cast<uint32_t>(view.axis[CONST3].dstStride),
                              static_cast<uint32_t>(view.axis[CONST4].dstStride)},
            .loopSize = {static_cast<uint32_t>(view.axis[CONST0].repeat),
                         static_cast<uint32_t>(view.axis[CONST1].repeat),
                         static_cast<uint32_t>(view.axis[CONST2].repeat),
                         static_cast<uint32_t>(view.axis[CONST3].repeat),
                         static_cast<uint32_t>(view.axis[CONST4].repeat)},
            .loopLpSize = {0, 0, 0, 0, 0},
            .loopRpSize = {0, 0, 0, 0, 0}};
        MultiCopyParams<T, CONST5> params = {copyLoopInfo, 0};
        for (uint64_t i = 0; i < view.axis[CONST7].repeat; i++) {
            for (uint64_t j = 0; j < view.axis[CONST6].repeat; j++) {
                for (uint64_t k = 0; k < view.axis[CONST5].repeat; k++) {
                    int64_t dstStride = i * view.axis[CONST7].dstStride + j * view.axis[CONST6].dstStride +
                                        k * view.axis[CONST5].dstStride;
                    int64_t srcStride = i * view.axis[CONST7].srcStride + j * view.axis[CONST6].srcStride +
                                        k * view.axis[CONST5].srcStride;
                    DataCopy<T, CONST5, config>(dst[dstStride], src[view.addr + srcStride], params);
                }
            }
        }
    }

    template <int32_t pos, class InnerPattern, class T, class V>
    __aicore__ inline void CopyInAux(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view)
    {
        if constexpr (CheckCopyIn<ReduceOp>()) {
            reduceOp_->template CopyIn<pos, InnerPattern, T>(dst, src, view);
        } else {
            if (view.isBlockAligned == 1U) {
                if constexpr (isContiguous) {
                    CopyInWithMoveAlign<pos, InnerPattern>(dst, src, view);
                } else {
                    CopyInWithMoveAlignNonContiguous<pos, InnerPattern>(dst, src, view);
                }
            } else {
                if constexpr (isContiguous) {
                    CopyInWithNddma<pos, InnerPattern>(dst, src, view);
                } else {
                    CopyInWithNddmaNonContiguous<pos, InnerPattern>(dst, src, view);
                }
            }
        }
    }

    // Alloc Tensor with bufId
    template <typename T, int32_t pos>
    __aicore__ inline LocalTensor<T> AllocTensorAux(uint8_t bufId)
    {
        int32_t offset = 0;
        if constexpr (pos < OpDag::ReduceOpPos) {
            constexpr uint8_t pongOffset = PRE_MTE2_NUM + PRE_MTE3_NUM + PRE_TEMP_CALC_NUM;
            if (bufId > pongOffset) {
                offset = PRE_MTE2_NUM * tiling_->basicBlock +
                         (PRE_MTE3_NUM + PRE_TEMP_CALC_NUM) * tiling_->basicBlock * Ratio();
            }
            int32_t newBufId = bufId > pongOffset ? bufId - pongOffset : bufId;
            if (newBufId < PRE_MTE2_NUM) {
                offset = offset + newBufId * tiling_->basicBlock;
            } else if (newBufId < PRE_MTE2_NUM + PRE_MTE3_NUM) {
                offset = offset + PRE_MTE2_NUM * tiling_->basicBlock +
                         (newBufId - PRE_MTE2_NUM) * tiling_->basicBlock * Ratio();
            } else {
                offset = offset + PRE_MTE2_NUM * tiling_->basicBlock + PRE_MTE3_NUM * tiling_->basicBlock * Ratio() +
                         (newBufId - PRE_MTE2_NUM - PRE_MTE3_NUM) * tiling_->basicBlock * Ratio();
            }
            RUN_LOG("Pre pos:%d AllocTensor ID: %d, offset: %d\n", pos, static_cast<int32_t>(bufId), offset);
            return preBufPool_[offset].template ReinterpretCast<T>();
        } else {
            offset = bufId * tiling_->resultBlock;
            RUN_LOG("Post pos:%d AllocTensor ID: %d, offset: %d\n", pos, static_cast<int32_t>(bufId), offset);
            return postBufPool_[offset].template ReinterpretCast<T>();
        }
    }

    template <typename U, int index>
    __aicore__ inline void SetVar(U value)
    {
        if constexpr (!__aux::IsSameType<OpDag, void>::Value) {
            OpScalars_.template Set<index>(value);
        }
    }

    template <typename ScalarType, typename ScalarValue>
    __aicore__ inline constexpr ScalarType GetScalar()
    {
        static_assert(!(Placeholder::IsVar<ScalarValue>::Value && Placeholder::IsInHolder<ScalarValue>::Value &&
                        Placeholder::IsConstValue<ScalarValue>::Value),
                      "The input parameter type is not FunBind, Var, Const or Holder.");
        if constexpr (Placeholder::IsVar<ScalarValue>::Value) {
            if constexpr (!__aux::IsSameType<OpDag, void>::Value) {
                return OpScalars_.template Get<ScalarValue::Pos>();
            } else {
                return 0;
            }
        } else if constexpr (Placeholder::IsInHolder<ScalarValue>::Value) {
            GlobalTensor<ScalarType> globalTensor;
            globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalarType*>(input_[ScalarValue::Pos].GetPhyAddr(0)));
            ScalarType scalar = globalTensor.GetValue(0);
            return scalar;
        } else if constexpr (Placeholder::IsConstValue<ScalarValue>::Value) {
            ScalarType scalar = static_cast<ScalarType>(ScalarValue::value);
            return scalar;
        }
    }

    template <class Pattern, class T, class S, class... Args>
    __aicore__ inline void ComputeAux(const LocalTensor<T>& src, S& shape, Args... args)
    {
        LocalTensor<T> dst = resBuf_.template ReinterpretCast<T>();
        reduceOp_->template Compute<Pattern, isBatchInvariant>(shape, dst, src);
    }

    template <class Pattern, class T, class S>
    __aicore__ inline void UpdateCacheAux(uint64_t rIndex, S& shape)
    {
        LocalTensor<T> src = resBuf_.template ReinterpretCast<T>();
        LocalTensor<T> cache = cache_.template ReinterpretCast<T>();
        reduceOp_->template UpdateCache<Pattern>(cache, src, rIndex, shape);
    }

    template <class T>
    __aicore__ inline void GetCacheAux(LocalTensor<T>& outTensor, int64_t offset)
    {
        LocalTensor<T> cache = cache_.template ReinterpretCast<T>();
        outTensor = cache[offset];
    }

    template <class T>
    __aicore__ inline LocalTensor<T> GetReduceResAux()
    {
        return resBuf_.template ReinterpretCast<T>();
    }

    template <class Pattern, typename InputT, class T, class S, class P>
    __aicore__ inline void PadValueAux(const LocalTensor<T>& ubTensor, S& shape, P& padding)
    {
        reduceOp_->template PadValue<Pattern, InputT>(ubTensor, shape, padding);
    }

    template <class T, class V>
    __aicore__ inline void CopyOutAux(const GlobalTensor<T>& dst, const LocalTensor<T>& src, V& view)
    {
        if constexpr (CheckCopyOut<ReduceOp>(0)) {
            reduceOp_->CopyOut(dst, src, view);
        } else {
            DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
            copyOutParams.blockCount = view.axis[1].repeat;
            copyOutParams.blockLen = view.axis[0].repeat * sizeof(T);
            copyOutParams.srcStride = (view.axis[1].srcStride - view.axis[0].repeat) * sizeof(T) / UB_BLOCK;
            DataCopyPad(dst[view.addr], src, copyOutParams);
        }
    }
};
} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops
#endif