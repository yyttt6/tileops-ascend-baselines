/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sch_batch_invariant.h
 * \brief aux for reduce schedule batch invariant (contiguous only), single-stage process when loopRCount != 0
 */

#ifndef _REDUCE_SCH_BATCH_INVARIANT_H_
#define _REDUCE_SCH_BATCH_INVARIANT_H_
#include "reduce_sch_aux_base.h"

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {
template <auto LoopInfo, class ReduceSch, bool IsStageOne, class OpDag = void>
struct ReduceSchAuxBatchInvariant
    : public ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                              ReduceSchAuxBatchInvariant<LoopInfo, ReduceSch, IsStageOne, OpDag>> {
    using AuxBase = ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                                     ReduceSchAuxBatchInvariant<LoopInfo, ReduceSch, IsStageOne, OpDag>>;
    using InDType = typename AuxBase::InDType;

public:
    constexpr static int32_t Dim = AuxBase::Dim;
    constexpr static uint64_t UB_BLOCK = AuxBase::UB_BLOCK;
    constexpr static uint64_t MIN_DTYPE_BYTES = AuxBase::MIN_DTYPE_BYTES;

public:
    struct IterAddr {
        uint64_t start = 0;
        uint64_t stride = 1;
    } iterAddr_[Dim];

public:
    __aicore__ inline ReduceSchAuxBatchInvariant(ReduceSch* sch, GlobalTensor<uint8_t>* input,
                                                 GlobalTensor<uint8_t>* output, GlobalTensor<uint8_t>* workspace,
                                                 const ReduceOpTilingData* tiling)
        : ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                           ReduceSchAuxBatchInvariant<LoopInfo, ReduceSch, IsStageOne, OpDag>>(sch, input, output,
                                                                                               workspace, tiling)
    {
        for (uint64_t i = 0; i < Dim; i++) {
            iterAddr_[i].stride = this->tiling_->shape[i];
        }
    }

    __aicore__ inline void SetLoopRange()
    {
        int32_t blockId = GetBlockIdx();
        if constexpr (IsBlockCutA<LoopInfo>()) {
            this->loopAStartIndex_ = blockId * this->tiling_->factorACntPerCore;
            this->loopAEndIndex_ = this->loopAStartIndex_ + this->tiling_->factorACntPerCore;
            if (unlikely(this->loopAEndIndex_ > this->tiling_->factorATotalCnt)) {
                this->loopAEndIndex_ = this->tiling_->factorATotalCnt;
            }
            constexpr int32_t aAxisIdx = LoopInfo->loopACount - 1;
            constexpr int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
            this->loopAAxisStep_ = Ops::Base::CeilDiv(this->tiling_->shape[aAxis], this->ubFactorA_);
            if constexpr (LoopInfo->loopInnerRCount > 0) {
                constexpr int32_t rAxisIdx = LoopInfo->loopInnerRCount - 1;
                constexpr int32_t rAxis = LoopInfo->loopInnerRAxis[rAxisIdx];
                this->loopRAxisStep_ = Ops::Base::CeilDiv(this->tiling_->shape[rAxis], this->ubFactorR_);
            }
        } else {
            this->loopAStartIndex_ = blockId / this->tiling_->groupR * this->tiling_->factorACntPerCore;
            this->loopAEndIndex_ = this->loopAStartIndex_ + this->tiling_->factorACntPerCore;
            if (unlikely(this->loopAEndIndex_ > this->tiling_->factorATotalCnt)) {
                this->loopAEndIndex_ = this->tiling_->factorATotalCnt;
            }
            if constexpr (LoopInfo->loopACount > 0) {
                constexpr int32_t aAxisIdx = LoopInfo->loopACount - 1;
                constexpr int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
                this->loopAAxisStep_ = Ops::Base::CeilDiv(this->tiling_->shape[aAxis], this->ubFactorA_);
            }

            this->loopRStartIndex_ = blockId / this->tiling_->groupR * this->tiling_->factorRTotalCnt +
                                     blockId % this->tiling_->groupR * this->tiling_->factorRCntPerCore;
            this->loopREndIndex_ = this->loopRStartIndex_ + this->tiling_->factorRCntPerCore;
            uint64_t maxRCnt = (blockId / this->tiling_->groupR + 1) * this->tiling_->factorRTotalCnt;
            uint64_t totalCnt = this->tiling_->factorATotalCnt * this->tiling_->factorRTotalCnt;
            maxRCnt = maxRCnt > totalCnt ? totalCnt : maxRCnt;
            if (unlikely(this->loopRStartIndex_ > maxRCnt)) {
                this->loopRStartIndex_ = maxRCnt;
            }
            if (unlikely(this->loopREndIndex_ > maxRCnt)) {
                this->loopREndIndex_ = maxRCnt;
            }
            if constexpr (LoopInfo->loopRCount > 0) {
                constexpr int32_t rAxisIdx = LoopInfo->loopRCount - 1;
                constexpr int32_t rAxis = LoopInfo->loopRAxis[rAxisIdx];
                this->loopRAxisStep_ = Ops::Base::CeilDiv(this->tiling_->shape[rAxis], this->ubFactorR_);
            }
        }
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        SetLoopRange();
        this->SetIterRRange();
        if constexpr (LoopInfo->loopRCount == 0) {
            for (uint64_t i = this->loopAStartIndex_; i < this->loopAEndIndex_; i++) {
                this->template CalculateIterA<LoopInfo->loopACount>(i);
                IterateInnerA<0, LoopInfo->loopInnerACount>(i - this->loopAStartIndex_, args...);
            }
        } else {
            if (this->loopRStartIndex_ == this->loopREndIndex_) {
                return;
            }
            for (uint64_t i = this->loopAStartIndex_; i < this->loopAEndIndex_; i++) {
                this->template CalculateIterA<LoopInfo->loopACount>(i);
                IterateInnerA<0, LoopInfo->loopInnerACount>(i - this->loopAStartIndex_, args...);
            }
        }
    }

    template <int32_t LoopAIdx>
    __aicore__ inline void CalculateIterA(uint64_t step)
    {
        if constexpr (LoopAIdx != 0) {
            constexpr auto axis = LoopInfo->loopAAxis[LoopAIdx - 1];
            if constexpr (LoopAIdx == LoopInfo->loopACount) {
                // 切分轴
                auto cur = step % this->loopAAxisStep_;
                iterAddr_[axis].start = cur * this->ubFactorA_;
                iterAddr_[axis].stride = this->tiling_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= this->ubFactorA_)) {
                    iterAddr_[axis].stride = this->ubFactorA_;
                }

                if constexpr (LoopAIdx > 0) {
                    this->CalculateIterA<LoopAIdx - 1>(step / this->loopAAxisStep_);
                }
            } else {
                iterAddr_[axis].start = step % this->tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                this->CalculateIterA<LoopAIdx - 1>(step / this->tiling_->shape[axis]);
            }
        }
    }

    template <int32_t start = 0, int32_t end = 0, class... Args>
    __aicore__ inline void IterateInnerA(int32_t aIndex, Args... args)
    {
        if constexpr (start == end) {
            int32_t startIdx = 0;
            if constexpr (LoopInfo->reduceDichotomy) {
                Shape<AuxBase::InnerPattern::Dim> shape = {.value = {0, 0}, .innerR = 0, .outerR = 0};
                SliceView<MAX_DIM> view;
                if constexpr (LoopInfo->loopRCount == 0) {
                    this->LinearComputeR(aIndex, startIdx, view, shape);
                    this->PostReduce(aIndex, view, shape);
                } else {
                    this->LinearComputeR(aIndex, startIdx, view, shape);
                    CopyOutGroup(view, shape);
                }
            }
        } else {
            constexpr int32_t axis = LoopInfo->loopInnerAAxis[start];
            uint64_t shapeSize = this->tiling_->sliceShape[axis];
            if constexpr (start + 1 == end) {
                uint64_t loopSize = shapeSize / this->ubFactorA_;
                uint64_t tail = shapeSize - loopSize * this->ubFactorA_;
                iterAddr_[axis].start = 0;
                iterAddr_[axis].stride = this->ubFactorA_;

                for (uint64_t i = 0; i < loopSize; i++) {
                    IterateInnerA<start + 1, end>(aIndex, args...);
                    iterAddr_[axis].start += this->ubFactorA_;
                }

                if (tail) {
                    iterAddr_[axis].stride = shapeSize - iterAddr_[axis].start;
                    IterateInnerA<start + 1, end>(aIndex, args...);
                }
            } else {
                for (uint64_t i = 0; i < shapeSize; i++) {
                    iterAddr_[axis].start = i;
                    IterateInnerA<start + 1, end>(aIndex, args...);
                }
            }
        }
    }

    template <int32_t LoopRIdx>
    __aicore__ inline void CalculateIterR(uint64_t step)
    {
        uint64_t temp = step;
        if constexpr (LoopRIdx != 0) {
            for (auto idx = LoopInfo->loopRCount - 1; idx >= LoopInfo->loopACount; --idx) {
                if (idx == LoopInfo->loopRCount - 1) {
                    constexpr auto axis = LoopInfo->loopRAxis[LoopInfo->loopRCount - 1];
                    auto cur = temp % this->loopRAxisStep_;
                    iterAddr_[axis].start = cur * this->ubFactorR_;
                    iterAddr_[axis].stride = this->tiling_->shape[axis] - iterAddr_[axis].start;
                    if (likely(iterAddr_[axis].stride >= this->ubFactorR_)) {
                        iterAddr_[axis].stride = this->ubFactorR_;
                    }
                    temp = temp / this->loopRAxisStep_;
                } else {
                    auto axis = LoopInfo->loopRAxis[idx];
                    iterAddr_[axis].start = temp % this->tiling_->shape[axis];
                    iterAddr_[axis].stride = 1;
                    temp = temp / this->tiling_->shape[axis];
                }
            }
        }
    }

    template <class V, class S>
    __aicore__ inline void CopyOutGroup(V& view, S& shape)
    {
        static_assert(IsSameType<typename AuxBase::PromoteDType, typename AuxBase::DataType>::value,
                      "group copy out dtype must be same with pre dag out.");
        SliceView<CONST2> newView;
        int32_t blockId = GetBlockIdx();
        uint64_t outLen = 1;
        uint64_t outRepeat = 1;
        this->CalculateOutView(view, shape, outLen, outRepeat);
        newView.axis[0].repeat = outLen;
        newView.axis[1].repeat = outRepeat;
        if constexpr (AuxBase::Pattern::TailA) {
            newView.axis[1].srcStride = Ops::Base::CeilAlign(outLen, UB_BLOCK / MIN_DTYPE_BYTES);
        } else {
            newView.axis[1].srcStride = outLen;
        }
        constexpr int32_t axis = AuxBase::Pattern::FirstA ? 0 : 1;
        uint64_t addrOffset = 0;
        for (int32_t i = axis; i < Dim; i += CONST2) {
            addrOffset += iterAddr_[i].start * this->tiling_->dstStride[i];
        }
        newView.addr = (blockId % this->tiling_->groupR) *
                           Ops::Base::CeilAlign(this->tiling_->outSize, static_cast<uint64_t>(AuxBase::VL_ELEMS)) +
                       addrOffset;

        SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LocalTensor<typename AuxBase::PromoteDType> outTensor;
        int64_t dimA = AuxBase::Pattern::TailA ? shape.value[1] : shape.value[0];
        int64_t dimAAlign = Ops::Base::CeilAlign(dimA, static_cast<int64_t>(AuxBase::VL_ELEMS));
        this->sch_->ReduceSch::template GetCacheAux(outTensor, (this->cacheCount_ - 1) * dimAAlign);
        GlobalTensor<typename AuxBase::PromoteDType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ typename AuxBase::PromoteDType*>(this->workspace_[0].GetPhyAddr(0)));
        this->sch_->ReduceSch::template CopyOutAux(globalTensor, outTensor, newView);
    }

    template <class V, class S, class P>
    __aicore__ inline void CalculatePatternARPadding(V& view, S& shape, P& padding)
    {
        const uint64_t blkSize = static_cast<uint64_t>(UB_BLOCK / sizeof(InDType));
        const uint64_t alignSize = view.isBlockAligned == 1U ? blkSize : 1UL;
        const bool isLastSplit = IsLoopSpliteRAxis<LoopInfo>(Dim - 1);
        const uint64_t mainBurstLen = isLastSplit ? this->ubFactorR_ : view.axis[0].repeat;
        const uint64_t busrtLen = view.axis[0].repeat;
        const uint64_t mainBurstLenAlign = Ops::Base::CeilAlign(mainBurstLen, alignSize);
        const uint64_t dimRAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(shape.value[1]), blkSize);
        int32_t burstPaddingRepeat = 1;
        int32_t rPaddingRepeat = 1;
        int32_t rPaddingStart = 1;
        int32_t aPaddingStart = 1;
        int32_t aPaddingTimes = 0;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                    rPaddingStart = view.axis[i].repeat;
                    break;
                } else {
                    burstPaddingRepeat *= view.axis[i].repeat;
                }
            }
        }
        // busrtLen补pad的起始位置
        padding.burstPaddingStart = view.axis[0].repeat;
        // busrtLen补pad长度为主块长度-尾块长度
        padding.burstPaddingLen = mainBurstLenAlign - busrtLen;
        // busrtLen补pad循环次数为R轴切分相同部分的大小
        padding.burstPaddingRepeat = burstPaddingRepeat * rPaddingStart;
        // R轴补pad的起始位置，即主块/尾块R轴切分的最后位置
        padding.rPaddingStart = rPaddingStart * burstPaddingRepeat * mainBurstLenAlign;
        // R轴pad的长度
        padding.rPaddingLen = dimRAlign - padding.rPaddingStart;
        // R轴补pad的循环次数，为A轴切分相同部分的大小
        // 因为以A轴为外循环，A轴大小在R轴循环过程中始终一致，每次A轴循环时，会重新计算shape大小
        padding.rPaddingRepeat = shape.value[0];
        // A轴补pad的起始位置，即主块/尾块A轴切分的最后位置
        padding.aPaddingStart = rPaddingRepeat * shape.value[1];
        // A轴pad的长度
        padding.aPaddingLen = 0;
        padding.aPaddingRepeat = 1;
    }

    template <class V, class S, class P>
    __aicore__ inline void CalculatePatternRAPadding(V& view, S& shape, P& padding)
    {
        const uint64_t blkSize = static_cast<uint64_t>(UB_BLOCK / sizeof(InDType));
        const uint64_t alignSize = view.isBlockAligned == 1U ? blkSize : 1UL;
        const uint64_t mainBurstLen = view.axis[0].repeat;
        const uint64_t busrtLen = view.axis[0].repeat;
        const uint64_t mainBurstLenAlign = Ops::Base::CeilAlign(mainBurstLen, alignSize);
        const uint64_t dimAAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(shape.value[1]), blkSize);
        int32_t burstPaddingRepeat = 1;
        int32_t aPaddingRepeat = 1;
        int32_t rPaddingStart = 1;
        int32_t rPaddingTimes = 0;
        int32_t aPaddingStart = 1;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                burstPaddingRepeat *= view.axis[i].repeat;
            }
        }
        padding.burstPaddingStart = view.axis[0].repeat;
        // busrtLen补pad长度为主块长度-尾块长度
        padding.burstPaddingLen = mainBurstLenAlign - busrtLen;
        // busrtLen补pad循环次数为A轴切分相同部分的大小
        padding.burstPaddingRepeat = burstPaddingRepeat * aPaddingStart;
        // A轴补pad的起始位置，即主块/尾块A轴切分的最后位置
        padding.aPaddingStart = aPaddingStart * burstPaddingRepeat * mainBurstLenAlign;
        // A轴pad的长度
        padding.aPaddingLen = dimAAlign - padding.aPaddingStart;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                    rPaddingStart = view.axis[i].repeat;
                    rPaddingTimes = this->ubFactorR_ - view.axis[i].repeat;
                    break;
                } else {
                    aPaddingRepeat *= view.axis[i].repeat;
                }
            }
        }
        // A轴补pad的循环次数，为R轴切分相同部分的大小，rPaddingStart=0时，表示R轴切分主块
        padding.aPaddingRepeat = aPaddingRepeat * rPaddingStart;
        // R轴补pad的起始位置，即主块/尾块R轴切分的最后位置
        padding.rPaddingStart = rPaddingStart * aPaddingRepeat * shape.value[1];
        // R轴pad的长度
        padding.rPaddingLen = rPaddingTimes * aPaddingRepeat * shape.value[1];
        padding.rPaddingRepeat = 1;
    }

    template <int32_t LoopInnerRIdx>
    __aicore__ inline void CalculateInnerIterR(uint64_t basicBlockIdx)
    {
        if constexpr (LoopInnerRIdx != 0) {
            constexpr auto axis = LoopInfo->loopInnerRAxis[LoopInnerRIdx - 1];
            if constexpr (LoopInnerRIdx == LoopInfo->loopInnerRCount) {
                auto cur = basicBlockIdx % this->loopRAxisStep_;
                iterAddr_[axis].start = cur * this->ubFactorR_;
                iterAddr_[axis].stride = this->tiling_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= this->ubFactorR_)) {
                    iterAddr_[axis].stride = this->ubFactorR_;
                }
                CalculateInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->loopRAxisStep_);
            } else {
                iterAddr_[axis].start = basicBlockIdx % this->tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalculateInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->tiling_->shape[axis]);
            }
        }
    }

    template <class V>
    __aicore__ inline void InitView(V& view)
    {
        constexpr static auto burstLenAxis = Dim - 1;
        view.axis[0].start = iterAddr_[burstLenAxis].start;
        view.axis[0].srcStride = 1;
        view.axis[0].repeat = GetBurstLen<LoopInfo, burstLenAxis>(iterAddr_, this->tiling_);
        view.axis[0].idx = burstLenAxis;
        view.axis[0].isAxisA = IsAxisA<AuxBase::Pattern::FirstA>(view.axis[0].idx);
        view.axisSize = 1;

        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 1; i < Dim; i++) {
                view.axisSize = i + 1;
                view.axis[i].start = iterAddr_[axis - 1].start;
                view.axis[i].repeat = GetRepeatStride<LoopInfo>(axis - 1, iterAddr_, this->tiling_,
                                                                view.axis[i].srcStride);
                view.axis[i].idx = axis - 1;
                view.axis[i].isAxisA = IsAxisA<AuxBase::Pattern::FirstA>(view.axis[i].idx);
                if (view.axis[i].idx <= 0) {
                    break;
                }
                axis = view.axis[i].idx;
            }
        }
        view.isBlockAligned = (this->tiling_->useNddma == 0);
    }

    template <class V, class S>
    __aicore__ inline void CalculateInnerShapeRAMode(V& view, S& shape)
    {
        int64_t rSize = 1;
        int64_t aSize = 1;
        if constexpr (AuxBase::Pattern::ID == PATTERN_A) {
            aSize = view.axis[0].repeat;
        }
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                // 第一个R轴的dstStride就是A轴大小
                aSize = (aSize == 1) ? view.axis[i].dstStride : aSize;
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                    rSize = rSize * this->ubFactorR_;
                } else {
                    rSize = rSize * view.axis[i].repeat;
                }
            }
        }

        shape.value[AuxBase::InnerPattern::Dim - 1] = aSize;
        shape.value[AuxBase::InnerPattern::Dim - 2] = rSize;
    }

    template <class T, class V>
    __aicore__ inline void CalculateViewARMode(V& view)
    {
        uint64_t alignedValue = (view.isBlockAligned == 1 ? (UB_BLOCK / sizeof(T)) : 1UL);
        int64_t value = Ops::Base::CeilAlign(view.axis[0].repeat, alignedValue);
        view.axis[0].dstStride = 1;
        if (IsLoopSpliteRAxis<LoopInfo>(Dim - 1)) {
            // R是切分轴时, 主尾块会先add起来, 所以R轴按照factorR算
            value = Ops::Base::CeilAlign(this->ubFactorR_, alignedValue);
        }
        // 因A轴循环会重新触发shape计算，所以A轴的shape不需要统一到ubFactorA,但是R轴shape需要统一到ubFactorR
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                    value = value * this->ubFactorR_;
                } else {
                    value = value * view.axis[i].repeat;
                }
            }
        }
        // nddma 整个R轴乘积block对齐
        if (view.isBlockAligned != 1) {
            value = Ops::Base::CeilAlign(static_cast<uint64_t>(value), UB_BLOCK / sizeof(T));
        }
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
            }
        }
    }

    template <class T, class V>
    __aicore__ inline void CalculateViewRAMode(V& view)
    {
        uint64_t alignedValue = (view.isBlockAligned == 1 ? (UB_BLOCK / MIN_DTYPE_BYTES) : 1UL);
        int64_t value = Ops::Base::CeilAlign(view.axis[0].repeat, alignedValue);
        view.axis[0].dstStride = 1;
        // 因A轴循环会重新触发shape计算，所以A轴的shape不需要统一到ubFactorA,但是R轴shape需要统一到ubFactorR
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
            }
        }
        // nddma 整个A轴乘积block对齐
        if (view.isBlockAligned != 1) {
            value = Ops::Base::CeilAlign(static_cast<uint64_t>(value), UB_BLOCK / sizeof(T));
        }
        if (value <= AuxBase::VL_ELEMS / 2) {
            value = Ops::Base::CeilAlign(value, static_cast<int64_t>(AuxBase::VL_ELEMS / 2)) + UB_BLOCK / sizeof(T);
        }
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                    value = value * this->ubFactorR_;
                } else {
                    value = value * view.axis[i].repeat;
                }
            }
        }
    }

    __aicore__ inline uint64_t CalculateCopyOutAddr(int32_t axis)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = axis; i < Dim; i += CONST2) {
            addrOffset += iterAddr_[i].start * this->tiling_->dstStride[i];
        }
        return addrOffset;
    }

    template <class T, class V>
    __aicore__ inline void CalculateInView(V& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < Dim; i++) {
            addrOffset += view.axis[i].start * view.axis[i].srcStride;
        }
        view.addr = addrOffset;
        if (view.axis[0].dstStride == 0) {
            this->template CalculateView<T>(view);
        }
    }
};
} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops
#endif