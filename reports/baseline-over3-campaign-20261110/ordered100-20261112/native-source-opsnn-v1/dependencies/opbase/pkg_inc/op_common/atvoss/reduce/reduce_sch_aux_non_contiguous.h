/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sch_aux_non_contiguous.h
 * \brief Non-contiguous version of reduce schedule aux
 */

#ifndef _REDUCE_SCH_AUX_NON_CONTIGUOUS_H_
#define _REDUCE_SCH_AUX_NON_CONTIGUOUS_H_

#include "reduce_sch_aux_base.h"

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {

template <auto LoopInfo, class ReduceSch, bool IsStageOne, class OpDag = void>
struct ReduceSchAuxNonContiguous
    : public ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                              ReduceSchAuxNonContiguous<LoopInfo, ReduceSch, IsStageOne, OpDag>> {
    using AuxBase = ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                                     ReduceSchAuxNonContiguous<LoopInfo, ReduceSch, IsStageOne, OpDag>>;
    using InDType = typename AuxBase::InDType;

public:
    constexpr static int32_t Dim = AuxBase::Dim;
    constexpr static uint64_t UB_BLOCK = AuxBase::UB_BLOCK;
    constexpr static uint64_t MIN_DTYPE_BYTES = AuxBase::MIN_DTYPE_BYTES;

public:
    uint64_t loopAAxisSliceStep_ = 0;
    uint64_t sliceShapeFactorA_ = 0;
    uint64_t sliceNumFactorA_ = 0;
    uint64_t loopRAxisSliceStep_ = 0;
    uint64_t sliceShapeFactorR_ = 0;
    uint64_t sliceNumFactorR_ = 0;

    struct IterAddr {
        uint64_t start = 0;
        uint64_t stride = 1;
        uint64_t sliceStart = 0;
        uint64_t sliceStride = 1;
    } iterAddr_[Dim];

public:
    __aicore__ inline ReduceSchAuxNonContiguous(ReduceSch* sch, GlobalTensor<uint8_t>* input,
                                                GlobalTensor<uint8_t>* output, GlobalTensor<uint8_t>* workspace,
                                                const ReduceOpTilingData* tiling)
        : ReduceSchAuxBase<LoopInfo, ReduceSch, IsStageOne, OpDag,
                           ReduceSchAuxNonContiguous<LoopInfo, ReduceSch, IsStageOne, OpDag>>(sch, input, output,
                                                                                              workspace, tiling)
    {
        for (uint64_t i = 0; i < Dim; i++) {
            iterAddr_[i].stride = tiling->sliceShape[i];
            iterAddr_[i].sliceStride = tiling->sliceNum[i];
        }
    }

    __aicore__ inline void CalcAxisStep(uint64_t sliceNum, uint64_t sliceShape, uint64_t ubFactor, uint64_t& axisStep,
                                        uint64_t& axisSliceStep)
    {
        if (ubFactor <= sliceShape) { // ub切在sliceShape
            axisStep = Ops::Base::CeilDiv(sliceShape, ubFactor);
            axisSliceStep = 1;
        } else { // ub切在sliceNum
            axisStep = 1;
            axisSliceStep = Ops::Base::CeilDiv(sliceNum, Ops::Base::CeilDiv(ubFactor, sliceShape));
        }
    }

    __aicore__ inline void ComputeIterAddress(int32_t axis, uint64_t step, uint64_t stepSize, uint64_t sliceStepSize,
                                              uint64_t sliceShapeFactor, uint64_t sliceNumFactor)
    {
        auto temp = step;
        auto curInSliceShape = temp % stepSize;
        temp /= stepSize;
        auto curInSliceNum = temp % sliceStepSize;
        iterAddr_[axis].start = curInSliceShape * sliceShapeFactor;
        iterAddr_[axis].sliceStart = curInSliceNum * sliceNumFactor;
        iterAddr_[axis].stride = this->tiling_->sliceShape[axis] - iterAddr_[axis].start;
        iterAddr_[axis].sliceStride = this->tiling_->shape[axis] / this->tiling_->sliceShape[axis] -
                                      iterAddr_[axis].sliceStart;
        if (likely(iterAddr_[axis].stride >= sliceShapeFactor)) {
            iterAddr_[axis].stride = sliceShapeFactor;
        }
        if (likely(iterAddr_[axis].sliceStride >= sliceNumFactor)) {
            iterAddr_[axis].sliceStride = sliceNumFactor;
        }
    }

    template <typename T>
    __aicore__ inline SliceFactors ComputeSliceFactors(T ubFactor, T sliceShape)
    {
        if (ubFactor <= sliceShape) {
            return {ubFactor, 1};
        } else {
            return {sliceShape, Ops::Base::CeilDiv(ubFactor, sliceShape)};
        }
    }

    __aicore__ inline void InitSliceFactors()
    {
        constexpr auto axisA = LoopInfo->loopAAxis[LoopInfo->loopACount - 1];
        auto factorsA = ComputeSliceFactors(this->ubFactorA_, this->tiling_->sliceShape[axisA]);
        sliceShapeFactorA_ = factorsA.sliceShapeFactor;
        sliceNumFactorA_ = factorsA.sliceNumFactor;

        if constexpr (LoopInfo->loopInnerRCount > 0) {
            constexpr auto axisR = LoopInfo->loopInnerRAxis[LoopInfo->loopInnerRCount - 1];
            auto factorsR = ComputeSliceFactors(this->ubFactorR_, this->tiling_->sliceShape[axisR]);
            sliceShapeFactorR_ = factorsR.sliceShapeFactor;
            sliceNumFactorR_ = factorsR.sliceNumFactor;
        } else if constexpr (LoopInfo->loopRCount > 0) {
            constexpr auto axisR = LoopInfo->loopRAxis[LoopInfo->loopRCount - 1];
            auto factorsR = ComputeSliceFactors(this->ubFactorR_, this->tiling_->sliceShape[axisR]);
            sliceShapeFactorR_ = factorsR.sliceShapeFactor;
            sliceNumFactorR_ = factorsR.sliceNumFactor;
        } else {
            sliceShapeFactorR_ = this->ubFactorR_;
            sliceNumFactorR_ = 1;
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
            CalcAxisStep(this->tiling_->sliceNum[aAxis], this->tiling_->sliceShape[aAxis], this->ubFactorA_,
                         this->loopAAxisStep_, loopAAxisSliceStep_);
            if constexpr (LoopInfo->loopInnerRCount > 0) {
                constexpr int32_t rAxisIdx = LoopInfo->loopInnerRCount - 1;
                constexpr int32_t rAxis = LoopInfo->loopInnerRAxis[rAxisIdx];
                CalcAxisStep(this->tiling_->sliceNum[rAxis], this->tiling_->sliceShape[rAxis], this->ubFactorR_,
                             this->loopRAxisStep_, loopRAxisSliceStep_);
            }
        } else {
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

            constexpr int32_t rAxisIdx = LoopInfo->loopRCount - 1;
            constexpr int32_t rAxis = LoopInfo->loopRAxis[rAxisIdx];
            CalcAxisStep(this->tiling_->sliceNum[rAxis], this->tiling_->sliceShape[rAxis], this->ubFactorR_,
                         this->loopRAxisStep_, loopRAxisSliceStep_);
            if constexpr (LoopInfo->loopACount > 0) {
                constexpr int32_t aAxisIdx = LoopInfo->loopACount - 1;
                constexpr int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
                CalcAxisStep(this->tiling_->sliceNum[aAxis], this->tiling_->sliceShape[aAxis], this->ubFactorA_,
                             this->loopAAxisStep_, loopAAxisSliceStep_);
            }
        }
        RUN_LOG("loopAStartIndex:%ld, loopAEndIndex:%ld, loopAAxisStep:%ld, loopAAxisSliceStep:%ld, ubFactorA:%ld, "
                "loopRStartIndex:%ld, loopREndIndex:%ld, loopRAxisStep:%ld, loopRAxisSliceStep:%ld, ubFactorR:%ld\n",
                this->loopAStartIndex_, this->loopAEndIndex_, this->loopAAxisStep_, loopAAxisSliceStep_,
                this->ubFactorA_, this->loopRStartIndex_, this->loopREndIndex_, this->loopRAxisStep_,
                loopRAxisSliceStep_, this->ubFactorR_);
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        InitSliceFactors();
        SetLoopRange();
        this->SetIterRRange();

        if constexpr (LoopInfo->loopRCount == 0) {
            for (uint64_t i = this->loopAStartIndex_; i < this->loopAEndIndex_; i++) {
                CalculateIterA<LoopInfo->loopACount>(i);
                this->template IterateInnerA<0, LoopInfo->loopInnerACount>(i - this->loopAStartIndex_, args...);
            }
        } else {
            if (this->loopRStartIndex_ == this->loopREndIndex_) {
                return;
            }
            this->template IterateInnerA<0, LoopInfo->loopInnerACount>(0, args...);
        }
    }

    template <int32_t LoopAIdx>
    __aicore__ inline void CalculateIterA(uint64_t step)
    {
        if constexpr (LoopAIdx != 0) {
            constexpr auto axis = LoopInfo->loopAAxis[LoopAIdx - 1];
            if constexpr (LoopAIdx == LoopInfo->loopACount) {
                ComputeIterAddress(axis, step, this->loopAAxisStep_, this->loopAAxisSliceStep_, sliceShapeFactorA_,
                                   sliceNumFactorA_);
                if constexpr (LoopAIdx > 0) {
                    CalculateIterA<LoopAIdx - 1>(step / this->loopAAxisStep_ / this->loopAAxisSliceStep_);
                }
            } else {
                iterAddr_[axis].start = step % this->tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalculateIterA<LoopAIdx - 1>(step / this->tiling_->shape[axis]);
            }
        }
    }

    template <class V, class S, class P>
    __aicore__ inline void CalculatePatternARPadding(V& view, S& shape, P& padding)
    {
        const uint64_t blkSize = static_cast<uint64_t>(UB_BLOCK / sizeof(InDType));
        const uint64_t alignSize = view.isBlockAligned == 1U ? blkSize : 1UL;
        const bool isLastSplit = IsLoopSpliteRAxis<LoopInfo>(Dim - 1);
        uint64_t mainBurstLen = 0;
        if (this->tiling_->sliceNum[Dim - 1] != 1) {
            if (this->ubFactorR_ <= this->tiling_->sliceShape[Dim - 1]) { // 切sliceShape
                mainBurstLen = isLastSplit ? this->ubFactorR_ : view.axis[0].repeat;
            } else {                                                      // 切sliceNum
                mainBurstLen = view.axis[0].repeat;
            }
        } else {
            mainBurstLen = isLastSplit ? this->ubFactorR_ : view.axis[0].repeat;
        }

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
                    rPaddingStart *= view.axis[i].repeat;
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
                auto axisIdx = view.axis[i].idx;
                if (IsLoopSpliteRAxis<LoopInfo>(axisIdx)) {
                    if (!view.axis[i].isSliceNum && this->ubFactorR_ < this->tiling_->sliceShape[axisIdx]) {
                        rPaddingStart = view.axis[i].repeat;
                        rPaddingTimes = this->ubFactorR_ - view.axis[i].repeat;
                    } else if (!view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        // tiling 切 sliceShape
                        aPaddingRepeat *= view.axis[i].repeat;
                    } else if (view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        // tiling 切 sliceNum
                        rPaddingStart = view.axis[i].repeat;
                        rPaddingTimes = this->ubFactorR_ / this->tiling_->sliceShape[axisIdx] - view.axis[i].repeat;
                    } else {
                        break;
                    }
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
                ComputeIterAddress(axis, basicBlockIdx, this->loopRAxisStep_, this->loopRAxisSliceStep_,
                                   sliceShapeFactorR_, sliceNumFactorR_);
                CalculateInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->loopRAxisStep_ /
                                                       this->loopRAxisSliceStep_);
            } else {
                iterAddr_[axis].start = basicBlockIdx % this->tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalculateInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->tiling_->shape[axis]);
            }
        }
    }

    template <int32_t LoopRIdx>
    __aicore__ inline void CalculateIterR(uint64_t step)
    {
        uint64_t temp = step;
        if constexpr (LoopRIdx != 0) {
            for (auto idx = LoopInfo->loopRCount - 1; idx > -1; --idx) {
                if (idx == LoopInfo->loopRCount - 1) {
                    constexpr auto axis = LoopInfo->loopRAxis[LoopInfo->loopRCount - 1];
                    ComputeIterAddress(axis, temp, this->loopRAxisStep_, this->loopRAxisSliceStep_, sliceShapeFactorR_,
                                       sliceNumFactorR_);
                    temp = temp / this->loopRAxisStep_;
                    temp = temp / this->loopRAxisSliceStep_;
                } else {
                    auto axis = LoopInfo->loopRAxis[idx];
                    if (IsLoopSpliteAAxis<LoopInfo>(axis)) {
                        // axis both in AAxis and RAxis
                        ComputeIterAddress(axis, temp, this->loopAAxisStep_, this->loopAAxisSliceStep_,
                                           sliceShapeFactorA_, sliceNumFactorA_);
                        temp = temp / this->loopAAxisStep_;
                        temp = temp / this->loopAAxisSliceStep_;
                    } else {
                        iterAddr_[axis].start = temp % this->tiling_->shape[axis];
                        iterAddr_[axis].stride = 1;
                        temp = temp / this->tiling_->shape[axis];
                    }
                }
            }
        }
    }

    template <class V>
    __aicore__ inline void InitView(V& view)
    {
        constexpr static auto burstLenAxis = Dim - 1; // 获取第一个循环轴
        view.axis[0].start = iterAddr_[burstLenAxis].start;
        view.axis[0].srcStride = this->tiling_->stride[burstLenAxis];
        view.axis[0].repeat = iterAddr_[burstLenAxis].stride;
        view.axis[0].idx = burstLenAxis;
        view.axis[0].isAxisA = IsAxisA<AuxBase::Pattern::FirstA>(view.axis[0].idx);
        view.axisSize = 1;
        if (this->tiling_->sliceNum[burstLenAxis] != 1) {
            view.axis[1].start = iterAddr_[burstLenAxis].sliceStart;
            view.axis[1].srcStride = this->tiling_->sliceStride[burstLenAxis];
            view.axis[1].repeat = iterAddr_[burstLenAxis].sliceStride;
            view.axis[1].idx = view.axis[0].idx;
            view.axis[1].isAxisA = view.axis[0].isAxisA;
            view.axis[1].isSliceNum = true;
            view.axisSize = 2;
        }
        int32_t axisStart = view.axisSize;

        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 1; i < Dim; i++) {
                if (this->tiling_->shape[axis - 1] == 1 && i != Dim - 1) {
                    axis = axis - 1;
                    continue;
                }
                int32_t currentIdx = axisStart;
                view.axis[currentIdx].start = iterAddr_[axis - 1].start;
                view.axis[currentIdx].srcStride = this->tiling_->stride[axis - 1];
                view.axis[currentIdx].repeat = iterAddr_[axis - 1].stride;
                view.axis[currentIdx].idx = axis - 1;
                view.axis[currentIdx].isAxisA = IsAxisA<AuxBase::Pattern::FirstA>(view.axis[currentIdx].idx);
                axisStart++;
                view.axisSize = axisStart;
                if (this->tiling_->sliceNum[burstLenAxis - i] != 1) {
                    view.axis[currentIdx + 1].start = iterAddr_[burstLenAxis - i].sliceStart;
                    view.axis[currentIdx + 1].srcStride = this->tiling_->sliceStride[burstLenAxis - i];
                    view.axis[currentIdx + 1].repeat = iterAddr_[burstLenAxis - i].sliceStride;
                    view.axis[currentIdx + 1].idx = view.axis[currentIdx].idx;
                    view.axis[currentIdx + 1].isAxisA = view.axis[currentIdx].isAxisA;
                    view.axis[currentIdx + 1].isSliceNum = true;
                    axisStart++;
                    view.axisSize = axisStart;
                }
                if (view.axis[currentIdx].idx <= 0) {
                    break;
                }
                axis = view.axis[currentIdx].idx;
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
                if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx) && !view.axis[i].isSliceNum) {
                    if (this->ubFactorR_ <= this->tiling_->sliceShape[view.axis[i].idx]) { // UB切在sliceShape
                        rSize = rSize * this->ubFactorR_;
                    } else {                                                               // UB切在sliceNum
                        rSize = rSize * view.axis[i].repeat;
                        rSize = rSize * (this->ubFactorR_ / view.axis[i].repeat);
                    }
                } else if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx) && view.axis[i].isSliceNum) {
                    // 这部分的rsize已经在 UB切在sliceNum 分支中计算过，跳过
                    continue;
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
            if (this->ubFactorR_ < this->tiling_->sliceShape[view.axis[0].idx]) {
                value = Ops::Base::CeilAlign(this->ubFactorR_, alignedValue);
            }
        }
        // 因A轴循环会重新触发shape计算，所以A轴的shape不需要统一到ubFactorA,但是R轴shape需要统一到ubFactorR
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                auto axisIdx = view.axis[i].idx;
                view.axis[i].dstStride = value;
                if (IsLoopSpliteRAxis<LoopInfo>(axisIdx)) {
                    if (!view.axis[i].isSliceNum && this->ubFactorR_ < this->tiling_->sliceShape[axisIdx]) {
                        value = value * this->ubFactorR_;
                    } else if (!view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        value = value * view.axis[i].repeat;
                    } else if (view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        value = value * this->ubFactorR_ / this->tiling_->sliceShape[axisIdx];
                    } else {
                        value = value * view.axis[i].repeat;
                    }
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
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                auto axisIdx = view.axis[i].idx;
                view.axis[i].dstStride = value;
                if (IsLoopSpliteRAxis<LoopInfo>(axisIdx)) {
                    if (!view.axis[i].isSliceNum && this->ubFactorR_ < this->tiling_->sliceShape[axisIdx]) {
                        value = value * this->ubFactorR_;
                    } else if (!view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        value = value * view.axis[i].repeat;
                    } else if (view.axis[i].isSliceNum && this->ubFactorR_ >= this->tiling_->sliceShape[axisIdx]) {
                        value = value * this->ubFactorR_ / this->tiling_->sliceShape[axisIdx];
                    } else {
                        value = value * view.axis[i].repeat;
                    }
                } else {
                    value = value * view.axis[i].repeat;
                }
            }
        }
    }

    __aicore__ inline uint64_t CalculateCopyOutAddr(int32_t axis)
    {
        uint64_t addrOffset = 0;
        auto aAxis = LoopInfo->loopAAxis[LoopInfo->loopACount - 1];
        for (int32_t i = axis; i < Dim; i += CONST2) {
            if (i == aAxis) {
                addrOffset += iterAddr_[i].start * this->tiling_->dstStride[i];
                if (this->tiling_->sliceNum[i] != 1) {
                    addrOffset += iterAddr_[i].sliceStart * this->tiling_->sliceShape[i];
                }
            } else {
                addrOffset += iterAddr_[i].start * this->tiling_->dstStride[i];
            }
        }
        return addrOffset;
    }

    __aicore__ inline uint64_t CalculateCopyOutGroupAddr(int32_t axis, int32_t innerA)
    {
        int32_t blockId = GetBlockIdx();
        uint64_t addrOffset = 0;
        if constexpr (LoopInfo->loopInnerACount > 0) {
            for (int32_t i = axis; i < Dim; i += CONST2) {
                addrOffset += iterAddr_[i].start * this->tiling_->dstStride[i];
            }
        }

        uint64_t axisStep = LoopInfo->loopACount > 0 ? this->loopAAxisStep_ : 1;
        uint64_t axisSliceStep = LoopInfo->loopACount > 0 ? this->loopAAxisSliceStep_ : 1;
        uint64_t step = axisStep * axisSliceStep;
        return (blockId % this->tiling_->groupR) *
                   Ops::Base::CeilAlign(this->tiling_->outSize, static_cast<uint64_t>(AuxBase::VL_ELEMS)) +
               (blockId / this->tiling_->groupR % step) * this->ubFactorA_ * innerA +
               (blockId / (this->tiling_->groupR * step)) * this->tiling_->shape[axis] * innerA + addrOffset;
    }

    template <class T, class V>
    __aicore__ inline void CalculateInView(V& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < view.axisSize; i++) {
            addrOffset += view.axis[i].start * view.axis[i].srcStride;
        }
        view.addr = addrOffset; // 搬运地址
        if (view.axis[0].dstStride == 0) {
            this->template CalculateView<T>(view);
        }
    }
};

} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops

#endif