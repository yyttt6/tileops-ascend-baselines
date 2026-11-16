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
 * \file reduce_operator.h
 * \brief reduce operator
 */

#ifndef _REDUCE_OPERATOR_H_
#define _REDUCE_OPERATOR_H_
#ifdef __CCE_AICORE__
#include "reduce_util.h"
#include "reduce_sch_aux.h"
#include "adv_api/reduce/reduce.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#endif
#include "atvoss/util/vec.h"

namespace Ops {
namespace Base {
namespace Vec {
#ifdef __CCE_AICORE__
using namespace ReduceOpTmpl;
inline constexpr MicroAPI::CastTrait INT_TO_FP_CAST_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};
inline constexpr MicroAPI::CastTrait FP_TO_INT_CAST_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
constexpr uint16_t VL_LEN = Ops::Base::GetVRegSize();
constexpr uint16_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint16_t BASE_FOLD = 4;
constexpr uint16_t BASE_FOLD_I64 = 3;
constexpr uint16_t FOLD1 = 1;
constexpr uint16_t FOLD2 = 2;
constexpr uint16_t FOLD3 = 3;
constexpr uint16_t MAX_OFFSET = 16;
constexpr int32_t REDUCE_OP_SUM = 1;
constexpr int32_t REDUCE_OP_PROD = 2;
constexpr int32_t REDUCE_OP_MIN = 3;
constexpr int32_t REDUCE_OP_MAX = 4;
constexpr uint32_t U16_STRIDE = 65535;
constexpr uint16_t REGULAR_FOLD_NUM = 2;

template <typename T>
struct Signed2Unsigned {
    using Type = T;
};

template <>
struct Signed2Unsigned<int32_t> {
    using Type = uint32_t;
};

template <>
struct Signed2Unsigned<int16_t> {
    using Type = uint16_t;
};

template <>
struct Signed2Unsigned<int64_t> {
    using Type = uint32_t;
};

template <typename T>
__aicore__ inline constexpr bool IsB64()
{
    return IsSameType<T, int64_t>::value || IsSameType<T, uint64_t>::value;
}

__aicore__ inline int32_t MainR(int64_t dimR, bool isAR, int32_t blkElems)
{
    int64_t mainR = 1;
    for (uint16_t i = 1; i < MAX_OFFSET; i++) {
        if ((dimR >> i) == 0) {
            break;
        }
        mainR = 1 << i;
    }
    if (isAR && dimR < static_cast<int64_t>(blkElems)) {
        mainR = dimR;
    }
    return static_cast<int32_t>(mainR);
}

__aicore__ inline uint16_t CalcFolds(int32_t base)
{
    uint16_t folds = 0;
    for (uint16_t i = 1; i < MAX_OFFSET; i++) {
        if ((base >> i) == 0) {
            break;
        }
        folds++;
    }
    return folds;
}

__aicore__ inline uint16_t FindClosestPowerOfTwo(uint32_t dimR)
{
    uint32_t r = 0;
    while (dimR > 1) {
        dimR >>= 1;
        r++;
    }
    return r;
}

#if defined(__NPU_ARCH__) &&                                                                                      \
        ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113)) || \
    defined(__ASC_NPU_HOST__)
__aicore__ constexpr inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1 || !defined(SPLIT_CORE_VEC)
    return (num1 + num2 - 1) / num2;
#else
    return get_repeat_ceiling(num1, num2);
#endif
}
#else
__aicore__ inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}
#endif

template <bool IsBlockPad, const AscendC::MicroAPI::RegTrait& Trait, typename InputT, typename T, class S, class P>
__aicore__ inline void PaddingARMode(__ubuf__ T* dstAddr, T padValue, S& shape, P& padding)
{
    const uint32_t dimR = shape.value[1];
    constexpr uint16_t dtypeSize = IsB64<T>() ? sizeof(float) : sizeof(T);
    constexpr uint64_t blkElems = BLOCK_SIZE / sizeof(InputT);
    constexpr uint64_t vLElems = VL_LEN / dtypeSize;

    const uint16_t burstPaddingRepeat = static_cast<uint16_t>(padding.burstPaddingRepeat);
    const uint16_t rPaddingRepeat = static_cast<uint16_t>(padding.rPaddingRepeat);
    const uint16_t aPaddingRepeat = static_cast<uint16_t>(padding.aPaddingRepeat);
    const uint64_t burstPaddingLen = padding.burstPaddingLen;
    const uint32_t rPaddingLen = static_cast<uint32_t>(padding.rPaddingLen);
    const uint32_t aPaddingLen = static_cast<uint32_t>(padding.aPaddingLen);
    const uint64_t burstPaddingStart = padding.burstPaddingStart;
    const uint64_t aPaddingStart = padding.aPaddingStart;
    const uint64_t rPaddingStart = padding.rPaddingStart;
    // int64_t mte2 move_align to 4, while reg align to 8
    const uint64_t mainBurstLenAlign = Ops::Base::CeilAlign(
        static_cast<uint64_t>(padding.burstPaddingStart + padding.burstPaddingLen), blkElems);
    // burstLen Padding的block下对齐
    const uint64_t burstPaddingStartAlign = burstPaddingStart / blkElems * blkElems;
    // burstLen Padding的block上对齐，如果burstPaddingStart本身对齐，burstPaddingStartAlign = burstPaddingStartCeilAlign
    const uint64_t burstPaddingStartCeilAlign = Ops::Base::CeilAlign(burstPaddingStart, blkElems);
    // 不做pad的长度，不能掩盖有效数据，burstPaddingStart非block对齐有效，否则为0
    const uint64_t noPadMask = burstPaddingStart - burstPaddingStartAlign;
    // 如果存在非对齐，第一次非对齐处理VLength长度
    const uint64_t burstPaddingStartUpVL = burstPaddingStartAlign + vLElems;
    // 是否需要对burstlen做block对齐的补pad
    const uint16_t isBlockPadding = (IsBlockPad && noPadMask > 0 && burstPaddingLen > 0) ? 1 : 0;
    // 如果存在非对齐，余下对齐部分的起始位置
    const uint64_t burstPaddingRemainingStart = (isBlockPadding > 0 ? burstPaddingStartUpVL :
                                                                      burstPaddingStartCeilAlign);
    // 如果存在非对齐，余下对齐部分的补pad长度
    const uint64_t burstPaddingRemainingLen = (mainBurstLenAlign > burstPaddingRemainingStart ?
                                                   (mainBurstLenAlign - burstPaddingRemainingStart) :
                                                   0);
    // 如果存在非对齐，余下对齐部分的repeat次数
    const uint16_t burstVLRemainingRepeat = Ops::Base::CeilDiv(static_cast<uint64_t>(burstPaddingRemainingLen),
                                                               static_cast<uint64_t>(vLElems));
    // r轴补pad的block下对齐
    const uint64_t rPaddingStartAlign = rPaddingStart / blkElems * blkElems;
    // r轴补pad的block上对齐
    const uint64_t rPaddingStartCeilAlign = Ops::Base::CeilAlign(rPaddingStart, blkElems);
    // 不做pad的长度，不能掩盖有效数据，rPaddingStart非block对齐有效，否则为0, nddma搬入可能非对齐
    const uint64_t noRPadMask = rPaddingStart - rPaddingStartAlign;
    // 如果存在非对齐，第一次非对齐处理VLength长度
    const uint64_t rPaddingStartUpVL = rPaddingStartAlign + vLElems;
    // r轴补pad后的长度
    const uint64_t rPaddingLenAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(rPaddingStart + rPaddingLen),
                                                           blkElems);
    // 是否需要对r轴做block对齐的补pad
    const uint16_t isRPaddingBlock = (rPaddingStart % blkElems == 0 ? 0 : 1);
    // 如果存在非对齐，r轴余下对齐部分的起始位置
    const uint64_t rPaddingRemainingStart = (rPaddingStart % blkElems == 0 ? rPaddingStartCeilAlign :
                                                                             rPaddingStartUpVL);
    // 如果存在非对齐，余下对齐部分的补pad长度
    const uint64_t rPaddingRemainingLen = (rPaddingLenAlign > rPaddingRemainingStart ?
                                               (rPaddingLenAlign - rPaddingRemainingStart) :
                                               0);
    // 如果存在非对齐，余下对齐部分的repeat次数
    const uint16_t rVlRepeat = Ops::Base::CeilDiv(static_cast<uint64_t>(rPaddingRemainingLen),
                                                  static_cast<uint64_t>(vLElems));
    const uint16_t aVlRepeat = Ops::Base::CeilDiv(static_cast<uint64_t>(aPaddingLen), static_cast<uint64_t>(vLElems));

    const uint16_t isBurstVLRemainingRepeat = burstVLRemainingRepeat > 0 ? 1 : 0;
    const uint16_t isBurstRepeat = burstPaddingRepeat > 0 ? 1 : 0;
    const uint16_t isRepeatA = aVlRepeat > 0 ? 1 : 0;
    const uint16_t isRepeatR = rVlRepeat > 0 ? 1 : 0;
    RUN_LOG("mainBurstLenAlign:%ld, burstPaddingStartAlign:%ld, burstPaddingStartCeilAlign:%ld, "
            "burstPaddingStartUpVL:%ld, burstPaddingRemainingStart:%ld, "
            "burstPaddingRemainingLen:%ld, rPaddingStartUpVL:%ld, rPaddingStartAlign:%ld, rPaddingStartCeilAlign:%ld, "
            "rPaddingLenAlign:%ld, rPaddingRemainingStart:%d, "
            "rPaddingRemainingLen:%ld\n",
            mainBurstLenAlign, burstPaddingStartAlign, burstPaddingStartCeilAlign, burstPaddingStartUpVL,
            burstPaddingRemainingStart, burstPaddingRemainingLen, rPaddingStartUpVL, rPaddingStartAlign,
            rPaddingStartCeilAlign, rPaddingLenAlign, rPaddingRemainingStart, rPaddingRemainingLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T, Trait> vreg;
        AscendC::MicroAPI::RegTensor<T, Trait> vregTmp;
        AscendC::MicroAPI::Duplicate(vreg, padValue);
        AscendC::MicroAPI::MaskReg mask;
        /**
         *      |-----------padMask-------------|--------PaddingLen--------|--------burstPaddingRemainingLen-----------|
         *  PaddingStartAlign(32B)        PaddingStart                  VLAlign(256B)                 mainBurstLenAlign
         */

        if constexpr (IsBlockPad) {
            for (uint16_t k = 0; k < isBlockPadding; k++) {
                AscendC::MicroAPI::RegTensor<T, Trait> vregPad;
                AscendC::MicroAPI::MaskReg maskDump;
                AscendC::MicroAPI::MaskReg
                    maskAll = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                uint32_t noPadLen = noPadMask;
                maskDump = AscendC::MicroAPI::UpdateMask<T, Trait>(noPadLen);
                AscendC::MicroAPI::MaskNot(maskDump, maskDump, maskAll);
                for (uint16_t i = 0; i < rPaddingRepeat; i++) {
                    for (uint16_t j = 0; j < burstPaddingRepeat; j++) {
                        // 32B Align + Copy In
                        AscendC::MicroAPI::DataCopy(
                            vregPad, dstAddr + burstPaddingStartAlign + i * dimR + j * mainBurstLenAlign);
                        // Dump
                        vregTmp = vregPad;
                        AscendC::MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::ZEROING, T>(vregTmp, padValue,
                                                                                                      maskDump);
                        AscendC::MicroAPI::Copy(vregPad, vregTmp, maskDump);
                        // Copy Out
                        uint32_t outLen = mainBurstLenAlign - burstPaddingStartAlign;
                        mask = AscendC::MicroAPI::UpdateMask<T, Trait>(outLen);
                        AscendC::MicroAPI::DataCopy(dstAddr + burstPaddingStartAlign + i * dimR + j * mainBurstLenAlign,
                                                    vregPad, mask);
                    }
                }
            }
        }

        for (uint16_t l = 0; l < isBurstVLRemainingRepeat; l++) {
            for (uint16_t i = 0; i < rPaddingRepeat; i++) {
                for (uint16_t j = 0; j < burstPaddingRepeat; j++) {
                    uint32_t scalar2 = burstPaddingRemainingLen;
                    for (uint16_t k = 0; k < burstVLRemainingRepeat; k++) {
                        mask = AscendC::MicroAPI::UpdateMask<T, Trait>(scalar2);
                        AscendC::MicroAPI::DataCopy(
                            dstAddr + burstPaddingRemainingStart + i * dimR + j * mainBurstLenAlign + k * vLElems, vreg,
                            mask);
                    }
                }
            }
        }

        for (uint16_t k = 0; k < isRPaddingBlock; k++) {
            AscendC::MicroAPI::RegTensor<T, Trait> vregPad;
            AscendC::MicroAPI::MaskReg maskDump;
            AscendC::MicroAPI::MaskReg
                maskAll = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            uint32_t noPadLen = noRPadMask;
            maskDump = AscendC::MicroAPI::UpdateMask<T, Trait>(noPadLen);
            AscendC::MicroAPI::MaskNot(maskDump, maskDump, maskAll);
            for (uint16_t i = 0; i < rPaddingRepeat; i++) {
                // 32B Align + Copy In
                AscendC::MicroAPI::DataCopy(vregPad, dstAddr + rPaddingStartAlign + i * dimR);
                // Dump
                vregTmp = vregPad;
                AscendC::MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::ZEROING, T>(vregTmp, padValue,
                                                                                              maskDump);
                AscendC::MicroAPI::Copy(vregPad, vregTmp, maskDump);
                // Copy Out
                uint32_t outLen = rPaddingLenAlign - rPaddingStartAlign;
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(outLen);
                AscendC::MicroAPI::DataCopy(dstAddr + rPaddingStartAlign + i * dimR, vregPad, mask);
            }
        }

        // R
        for (uint16_t k = 0; k < isRepeatR; k++) {
            for (uint16_t i = 0; i < rPaddingRepeat; i++) {
                uint32_t maskR = rPaddingRemainingLen;
                for (uint16_t j = 0; j < rVlRepeat; j++) {
                    mask = AscendC::MicroAPI::UpdateMask<T, Trait>(maskR);
                    AscendC::MicroAPI::DataCopy(dstAddr + rPaddingRemainingStart + i * dimR + j * vLElems, vreg, mask);
                }
            }
        }
        // A
        for (uint16_t k = 0; k < isRepeatA; k++) {
            for (uint16_t i = 0; i < aPaddingRepeat; i++) {
                uint32_t maskA = aPaddingLen;
                for (uint16_t j = 0; j < aVlRepeat; j++) {
                    mask = AscendC::MicroAPI::UpdateMask<T, Trait>(maskA);
                    AscendC::MicroAPI::DataCopy(dstAddr + aPaddingStart + i * aPaddingLen + j * vLElems, vreg, mask);
                }
            }
        }
    }
}

template <const AscendC::MicroAPI::RegTrait& Trait, typename InputT, typename T, class S, class P>
__aicore__ inline void PaddingRAMode(__ubuf__ T* dstAddr, T padValue, S& shape, P& padding)
{
    constexpr uint16_t dtypeSize = IsB64<T>() ? sizeof(float) : sizeof(T);
    constexpr uint64_t blkElems = BLOCK_SIZE / sizeof(InputT);
    constexpr uint64_t vLElems = VL_LEN / dtypeSize;
    const uint64_t dimA = shape.value[1];
    const uint16_t rPaddingRepeat = static_cast<uint16_t>(padding.rPaddingRepeat);
    const uint16_t aPaddingRepeat = static_cast<uint16_t>(padding.aPaddingRepeat);
    const uint64_t rPaddingLen = padding.rPaddingLen;
    const uint64_t aPaddingLen = padding.aPaddingLen;
    const uint64_t aPaddingStart = padding.aPaddingStart;
    const uint64_t rPaddingStart = padding.rPaddingStart;

    // a轴补pad的block下对齐
    const uint64_t aPaddingStartAlign = aPaddingStart / blkElems * blkElems;
    // a轴补pad的block上对齐
    const uint64_t aPaddingStartCeilAlign = Ops::Base::CeilAlign(aPaddingStart, blkElems);
    // 不做pad的长度，不能掩盖有效数据，aPaddingStart非block对齐有效，否则为0, nddma搬入可能非对齐
    const uint64_t noAPadMask = aPaddingStart - aPaddingStartAlign;
    // 如果存在非对齐，第一次非对齐处理VLength长度
    const uint64_t aPaddingStartUpVL = aPaddingStartAlign + vLElems;
    // a轴补pad后的长度
    const uint64_t aPaddingLenAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(aPaddingStart + aPaddingLen),
                                                           blkElems);
    // 是否需要对a轴做block对齐的补pad
    const uint16_t isAPaddingBlock = (aPaddingStart % blkElems == 0 ? 0 : 1);
    // 如果存在非对齐，a轴余下对齐部分的起始位置
    const uint64_t aPaddingRemainingStart = (aPaddingStart % blkElems == 0 ? aPaddingStartCeilAlign :
                                                                             aPaddingStartUpVL);
    // 如果存在非对齐，余下对齐部分的补pad长度
    const uint64_t aPaddingRemainingLen = (aPaddingLenAlign > aPaddingRemainingStart ?
                                               (aPaddingLenAlign - aPaddingRemainingStart) :
                                               0);
    // 如果存在非对齐，余下对齐部分的repeat次数
    const uint16_t aVlRepeat = Ops::Base::CeilDiv(static_cast<uint64_t>(aPaddingRemainingLen),
                                                  static_cast<uint64_t>(vLElems));

    const uint16_t rVlRepeat = Ops::Base::CeilDiv(static_cast<uint64_t>(rPaddingLen), static_cast<uint64_t>(vLElems));
    const uint16_t isRepeatA = aVlRepeat > 0 ? 1 : 0;
    const uint16_t isRepeatR = rVlRepeat > 0 ? 1 : 0;
    RUN_LOG("aPaddingStartUpVL:%ld, aPaddingLenAlign:%ld, aPaddingRemainingStart:%d, "
            "aPaddingRemainingLen:%ld\n",
            aPaddingStartUpVL, aPaddingLenAlign, aPaddingRemainingStart, aPaddingRemainingLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T, Trait> vreg;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::Duplicate(vreg, padValue);
        // R
        for (uint16_t k = 0; k < isRepeatR; k++) {
            for (uint16_t i = 0; i < rPaddingRepeat; i++) {
                uint32_t maskR = rPaddingLen;
                for (uint16_t j = 0; j < rVlRepeat; j++) {
                    mask = AscendC::MicroAPI::UpdateMask<T, Trait>(maskR);
                    AscendC::MicroAPI::DataCopy(dstAddr + rPaddingStart + i * rPaddingLen + j * vLElems, vreg, mask);
                }
            }
        }
    }
}

template <const AscendC::MicroAPI::RegTrait& Trait, bool IsBlockPad, class Pattern, typename InputT, typename T,
          class S, class P>
__aicore__ inline void DoPadding(__ubuf__ T* dstAddr, T padValue, S& shape, P& padding)
{
    if constexpr (Pattern::TailA) {
        PaddingRAMode<Trait, InputT, T>(dstAddr, padValue, shape, padding);
    } else {
        PaddingARMode<IsBlockPad, Trait, InputT, T>(dstAddr, padValue, shape, padding);
    }
}

template <class Pattern, typename T, class S, int32_t Operator>
__aicore__ inline void DoCaching(const LocalTensor<T>& ubDst, const LocalTensor<T>& ubSrc, int64_t index, S& shape)
{
    constexpr int64_t vLElems = Ops::Base::GetVRegSize() / sizeof(T);
    const int64_t cacheID = GetCacheID(index);
    const int64_t dimA = Pattern::TailA ? shape.value[1] : shape.value[0];
    const int64_t stride = Ops::Base::CeilAlign(dimA, vLElems);
    // count A轴的大小 * vLElems
    const uint16_t outerLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(vLElems));
    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);
    const uint32_t innerLoopStride = static_cast<uint32_t>(stride); // cacahe的每一个idex的块的大小， A轴的大小
    LocalTensor<T> dstTensor = ubDst;
    LocalTensor<T> srcTensor = ubSrc;

    __VEC_SCOPE__
    {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* cah = (__local_mem__ T*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(dimA);
        AscendC::MicroAPI::RegTensor<T> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) { // outerLoopTimes是dimA的大小
            pMask = AscendC::MicroAPI::UpdateMask<T>(sreg);
            AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ T*)src + i * vLElems);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::MicroAPI::DataCopy(bReg, (__local_mem__ T*)dst + i * vLElems + j * innerLoopStride);
                if constexpr (Operator == REDUCE_OP_SUM) {
                    AscendC::MicroAPI::Add<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_PROD) {
                    AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_MIN) {
                    AscendC::MicroAPI::Min<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_MAX) {
                    AscendC::MicroAPI::Max<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
            }
            AscendC::MicroAPI::DataCopy((__local_mem__ T*)cah + i * vLElems, aReg, pMask);
        }
    }
}
#endif

template <typename PromteT>
class ReduceMaxOp : public ReduceOp<PromteT> {
public:
    __aicore__ inline ReduceMaxOp() {}
#ifdef __CCE_AICORE__
    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
        ReduceMax<PromteT, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);
    }

    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::RA>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
        ReduceMax<PromteT, AscendC::Pattern::Reduce::RA, true>(dst, src, srcShape, false);
    }

    template <class Pattern, class S>
    __aicore__ inline void UpdateCache(const LocalTensor<PromteT>& ubDst, const LocalTensor<PromteT>& ubSrc,
                                       int64_t index, S& shape)
    {
        DoCaching<Pattern, PromteT, S, REDUCE_OP_MAX>(ubDst, ubSrc, index, shape);
    }

    __aicore__ inline void BisectionPreHandle(const LocalTensor<PromteT>& dst, const LocalTensor<PromteT>& src0,
                                              const LocalTensor<PromteT>& src1, const int32_t& calCount)
    {
        AscendC::Max(dst, src0, src1, calCount);
    }

    template <class Pattern, typename InputT, class S, class P>
    __aicore__ inline void PadValue(const LocalTensor<PromteT>& dst, S& shape, P& padding)
    {
        PromteT padValue = GetPaddingValue<PromteT>();
        if constexpr (IsB64<PromteT>()) {
            DoPadding<AscendC::MicroAPI::RegTraitNumTwo, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        } else {
            DoPadding<AscendC::MicroAPI::RegTraitNumOne, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        }
    }

    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        U paddingValue = 1;
        if constexpr (IsSameType<U, uint8_t>::value) {
            paddingValue = UINT8_MIN_VALUE;
        } else if constexpr (IsSameType<U, int8_t>::value) {
            paddingValue = INT8_MIN_VALUE;
        } else if constexpr (IsSameType<U, half>::value) {
            paddingValue = HALF_MIN_VALUE;
        } else if constexpr (IsSameType<U, bfloat16_t>::value) {
            paddingValue = BFLOAT16_MIN_VALUE;
        } else if constexpr (IsSameType<U, float>::value) {
            paddingValue = FLOAT_MIN_VALUE;
        } else if constexpr (IsSameType<U, int32_t>::value) {
            paddingValue = INT32_MIN;
        } else if constexpr (IsSameType<U, int64_t>::value) {
            paddingValue = INT64_MIN;
        }
        return paddingValue;
    }
#endif
};

template <typename PromteT>
class ReduceMinOp : public ReduceOp<PromteT> {
public:
    __aicore__ inline ReduceMinOp() {}
#ifdef __CCE_AICORE__
    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
        ReduceMin<PromteT, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);
    }

    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::RA>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
        ReduceMin<PromteT, AscendC::Pattern::Reduce::RA, true>(dst, src, srcShape, false);
    }

    template <class Pattern, class S>
    __aicore__ inline void UpdateCache(const LocalTensor<PromteT>& ubDst, const LocalTensor<PromteT>& ubSrc,
                                       int64_t index, S& shape)
    {
        DoCaching<Pattern, PromteT, S, REDUCE_OP_MIN>(ubDst, ubSrc, index, shape);
    }

    __aicore__ inline void BisectionPreHandle(const LocalTensor<PromteT>& dst, const LocalTensor<PromteT>& src0,
                                              const LocalTensor<PromteT>& src1, const int32_t& calCount)
    {
        AscendC::Min(dst, src0, src1, calCount);
    }

    template <class Pattern, typename InputT, class S, class P>
    __aicore__ inline void PadValue(const LocalTensor<PromteT>& dst, S& shape, P& padding)
    {
        PromteT padValue = GetPaddingValue<PromteT>();
        if constexpr (IsB64<PromteT>()) {
            DoPadding<AscendC::MicroAPI::RegTraitNumTwo, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        } else {
            DoPadding<AscendC::MicroAPI::RegTraitNumOne, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        }
    }

    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        U paddingValue = 1;
        if constexpr (IsSameType<U, uint8_t>::value) {
            paddingValue = UINT8_MAX_VALUE;
        } else if constexpr (IsSameType<U, int8_t>::value) {
            paddingValue = INT8_MAX_VALUE;
        } else if constexpr (IsSameType<U, half>::value) {
            paddingValue = HALF_MAX_VALUE;
        } else if constexpr (IsSameType<U, bfloat16_t>::value) {
            paddingValue = BFLOAT16_MAX_VALUE;
        } else if constexpr (IsSameType<U, float>::value) {
            paddingValue = FLOAT_MAX_VALUE;
        } else if constexpr (IsSameType<U, int32_t>::value) {
            paddingValue = INT32_MAX;
        } else if constexpr (IsSameType<U, int64_t>::value) {
            paddingValue = INT64_MAX;
        } else if constexpr (IsSameType<U, int16_t>::value) {
            paddingValue = INT16_MAX;
        }
        return paddingValue;
    }
#endif
};

template <typename PromteT>
class ReduceProdOp : public ReduceOp<PromteT> {
public:
    __aicore__ inline ReduceProdOp() {}
#ifdef __CCE_AICORE__
    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        constexpr uint16_t blkElems = IsB64<PromteT>() ? BLOCK_SIZE / sizeof(float) : BLOCK_SIZE / sizeof(PromteT);
        int32_t mainR = MainR(shape.value[1], true, blkElems);
        if constexpr (IsB64<PromteT>()) {
            ReduceARI64<int64_t, AscendC::MicroAPI::RegTraitNumTwo>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                    (__ubuf__ PromteT*)src.GetPhyAddr(), shape.value[0],
                                                                    shape.value[1], mainR, shape.outerR, shape.innerR);
        } else if constexpr (IsSameType<PromteT, int32_t>::value || IsSameType<PromteT, uint32_t>::value ||
                             IsSameType<PromteT, float>::value) {
            ReduceAR<int32_t, AscendC::MicroAPI::RegTraitNumOne>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                 (__ubuf__ PromteT*)src.GetPhyAddr(), shape.value[0],
                                                                 shape.value[1], mainR, shape.outerR, shape.innerR);
        } else {
            ReduceAR<int16_t, AscendC::MicroAPI::RegTraitNumOne>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                 (__ubuf__ PromteT*)src.GetPhyAddr(), shape.value[0],
                                                                 shape.value[1], mainR, shape.outerR, shape.innerR);
        }
    }

    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::RA>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        constexpr uint16_t blkElems = IsB64<PromteT>() ? BLOCK_SIZE / sizeof(float) : BLOCK_SIZE / sizeof(PromteT);
        int32_t mainR = MainR(shape.value[0], false, blkElems);
        if constexpr (IsB64<PromteT>()) {
            ReduceRAI64<AscendC::MicroAPI::RegTraitNumTwo>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                           (__ubuf__ PromteT*)src.GetPhyAddr(), shape.value[1],
                                                           shape.value[0], mainR);
        } else {
            ReduceRA<AscendC::MicroAPI::RegTraitNumOne>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                        (__ubuf__ PromteT*)src.GetPhyAddr(), shape.value[1],
                                                        shape.value[0], mainR);
        }
    }

    template <typename DIndex, typename DCast, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceARLessBlock(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA,
                                             int32_t dimR, int32_t realR)
    {
        constexpr uint16_t vLElems = IsB64<PromteT>() ? VL_LEN / sizeof(float) : VL_LEN / sizeof(PromteT);
        const uint16_t loopA = Ops::Base::CeilDiv(dimA, static_cast<int32_t>(vLElems));
        const uint16_t innerLoopR = dimR - 1;
        uint32_t processA = dimA;
        float reciDimR = (float)1.0f / static_cast<float>(dimR);
        float reciDimA = (float)1.0f / static_cast<float>(dimA);
        int32_t dimRFloorAlign = realR * sizeof(PromteT) / 32 * 32 / sizeof(PromteT);
        uint32_t noPaddingLen = realR - dimRFloorAlign;
        uint32_t paddingLen = dimR - realR;
        uint32_t totalPaddingLen = noPaddingLen + paddingLen;
        int64_t blockElems = vLElems * dimR;

        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<DCast, Trait> vregReciDimR; // 1 / dimR
            AscendC::MicroAPI::RegTensor<DCast, Trait> vregReciDimA; // 1 / dimA
            AscendC::MicroAPI::RegTensor<DIndex, Trait> vregTmp0;
            AscendC::MicroAPI::RegTensor<DIndex, Trait> vregTmp1;    // idx0 % dimA * dimR
            AscendC::MicroAPI::RegTensor<DIndex, Trait> vregTmp2;    // idx0 % dimA
            AscendC::MicroAPI::RegTensor<DCast, Trait> vregCastFloat;
            AscendC::MicroAPI::RegTensor<DIndex, Trait> idx0;
            AscendC::MicroAPI::RegTensor<DIndex, Trait> idx1;
            AscendC::MicroAPI::RegTensor<DCast, Trait> idxCastFloat;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<DIndex>();
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregAllOne;
            AscendC::MicroAPI::MaskReg paddingMask1;
            AscendC::MicroAPI::MaskReg paddingMask2;
            AscendC::MicroAPI::Duplicate(vregAllOne, CONST1);

            if constexpr (IsB64<PromteT>()) {
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(processA);
                paddingMask1 = AscendC::MicroAPI::UpdateMask<int32_t>(noPaddingLen);
                paddingMask2 = AscendC::MicroAPI::UpdateMask<int32_t>(totalPaddingLen);
            } else {
                mask = AscendC::MicroAPI::UpdateMask<DIndex>(processA);
                paddingMask1 = AscendC::MicroAPI::UpdateMask<DIndex>(noPaddingLen);
                paddingMask2 = AscendC::MicroAPI::UpdateMask<DIndex>(totalPaddingLen);
            }
            AscendC::MicroAPI::MaskXor(paddingMask1, paddingMask1, paddingMask2, maskAll);
            // padding unAlign part fo dimR to 1
            for (uint16_t i = 0; i < (uint16_t)dimA; i++) {
                auto srcPaddingAddr = srcAddr + i * dimR + dimRFloorAlign;
                DataCopy(srcPaddingAddr, vregAllOne, paddingMask1);
            }

            // idx caculate = (idx // dimA) * (dimA * dimR) + (idx % dimA) * dimR + (k + idx % dimA) % dimR
            AscendC::MicroAPI::Duplicate(vregReciDimR, reciDimR);
            AscendC::MicroAPI::Duplicate(vregReciDimA, reciDimA);
            AscendC::MicroAPI::Arange(idx0, 0); // [0,1,2,3,4,5,6...]

            // idx0 % dimA
            AscendC::MicroAPI::Cast<DCast, DIndex, INT_TO_FP_CAST_TRAIT>(idxCastFloat, idx0, mask);
            AscendC::MicroAPI::Mul(vregCastFloat, idxCastFloat, vregReciDimA, mask);
            AscendC::MicroAPI::Cast<DIndex, DCast, FP_TO_INT_CAST_TRAIT>(vregTmp1, vregCastFloat, mask);
            AscendC::MicroAPI::Muls(vregTmp1, vregTmp1, dimA, mask);
            AscendC::MicroAPI::Sub(vregTmp2, idx0, vregTmp1, mask);

            // idx0 % dimA % dimR
            AscendC::MicroAPI::Cast<DCast, DIndex, INT_TO_FP_CAST_TRAIT>(vregCastFloat, vregTmp2, mask);
            AscendC::MicroAPI::Mul(vregCastFloat, vregCastFloat, vregReciDimR, mask);
            AscendC::MicroAPI::Cast<DIndex, DCast, FP_TO_INT_CAST_TRAIT>(vregTmp0, vregCastFloat, mask);
            AscendC::MicroAPI::Muls(vregTmp0, vregTmp0, dimR, mask);
            AscendC::MicroAPI::Sub(vregTmp0, vregTmp2, vregTmp0, mask);

            AscendC::MicroAPI::Muls(vregTmp1, vregTmp2, dimR, mask); // idx0 % dimA * dimR
            AscendC::MicroAPI::Add(idx0, vregTmp0, vregTmp1, mask);  // idx0 % dimA % dimR + idx0 % dimA * dimR

            for (uint16_t i = 0; i < loopA; i++) {
                AscendC::MicroAPI::DataCopyGather(
                    vreg0, srcAddr, (AscendC::MicroAPI::RegTensor<typename Signed2Unsigned<DIndex>::Type>&)idx0, mask);
                for (uint16_t j = 0; j < innerLoopR; j++) {
                    // k + idx % dimA
                    AscendC::MicroAPI::Adds(idx1, vregTmp2, (j + 1), mask);
                    // (k + idx % dimA) % dimR
                    AscendC::MicroAPI::Cast<DCast, DIndex, INT_TO_FP_CAST_TRAIT>(vregCastFloat, vregTmp0, mask);
                    AscendC::MicroAPI::Cast<DCast, DIndex, INT_TO_FP_CAST_TRAIT>(idxCastFloat, idx1, mask);
                    AscendC::MicroAPI::Mul(vregCastFloat, idxCastFloat, vregReciDimR, mask);
                    AscendC::MicroAPI::Cast<DIndex, DCast, FP_TO_INT_CAST_TRAIT>(vregTmp0, vregCastFloat, mask);
                    AscendC::MicroAPI::Muls(vregTmp0, vregTmp0, dimR, mask);
                    AscendC::MicroAPI::Sub(idx1, idx1, vregTmp0, mask);
                    // blockElems + (idx % dimA) * dimR + (k + idx % dimA) % dimR
                    AscendC::MicroAPI::Add(idx1, idx1, vregTmp1, mask);
                    AscendC::MicroAPI::Adds(idx1, idx1, i * blockElems, mask);
                    AscendC::MicroAPI::DataCopyGather(
                        vreg1, srcAddr, (AscendC::MicroAPI::RegTensor<typename Signed2Unsigned<DIndex>::Type>&)idx1,
                        mask);
                    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                }
                AscendC::MicroAPI::Adds(idx0, idx0, blockElems, mask);
                AscendC::MicroAPI::DataCopy((__ubuf__ PromteT*&)dstAddr + i * vLElems, vreg0, mask);
            }
        }
    }

    template <typename DIndex, typename DCast, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceARLessBlockI64(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA,
                                                int32_t dimR, int32_t realR)
    {
        constexpr uint16_t vLElems = IsB64<PromteT>() ? VL_LEN / sizeof(float) : VL_LEN / sizeof(PromteT);
        const uint16_t loopA = Ops::Base::CeilDiv(dimA, static_cast<int32_t>(vLElems));
        const uint16_t innerLoopR = realR - 1;
        uint32_t processA = dimA;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<DIndex, Trait> idx0;
            AscendC::MicroAPI::RegTensor<DIndex, Trait> idx1;
            AscendC::MicroAPI::MaskReg mask;
            if constexpr (IsB64<PromteT>()) {
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(processA);
            } else {
                mask = AscendC::MicroAPI::UpdateMask<DIndex>(processA);
            }
            for (uint16_t i = 0; i < loopA; i++) {
                DIndex startIdx = i * vLElems;
                AscendC::MicroAPI::Arange(idx0, startIdx);
                AscendC::MicroAPI::Muls(idx0, idx0, static_cast<DIndex>(dimR), mask);
                AscendC::MicroAPI::DataCopyGather(
                    vreg0, srcAddr, (AscendC::MicroAPI::RegTensor<typename Signed2Unsigned<DIndex>::Type>&)idx0, mask);
                for (uint16_t j = 0; j < innerLoopR; j++) {
                    AscendC::MicroAPI::Adds(idx1, idx0, (j + 1), mask);
                    AscendC::MicroAPI::DataCopyGather(
                        vreg1, srcAddr, (AscendC::MicroAPI::RegTensor<typename Signed2Unsigned<DIndex>::Type>&)idx1,
                        mask);
                    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                }
                AscendC::MicroAPI::DataCopy((__ubuf__ PromteT*&)dstAddr + i * vLElems, vreg0, mask);
            }
        }
    }

    template <typename T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceARFoldBase(
        __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const int32_t dimR, uint32_t maskElems, uint16_t blkElems,
        const uint16_t vLElems, const uint16_t loopANum, uint16_t fold1, uint16_t fold2, uint16_t fold3,
        uint16_t isBlkRepeat, uint16_t blkRepeats, AscendC::MicroAPI::MaskReg& mask,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg0, AscendC::MicroAPI::RegTensor<T, Trait>& vreg1,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg2, AscendC::MicroAPI::RegTensor<T, Trait>& vreg3,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg4, AscendC::MicroAPI::RegTensor<T, Trait>& vreg5,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg6, AscendC::MicroAPI::RegTensor<T, Trait>& vreg7)
    {
        for (uint16_t i = 0; i < fold1; i++) {
            mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + vLElems + loopA * dimR);
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < fold2; i++) {
            mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + vLElems + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg2, srcAddr + vLElems * 2 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg3, srcAddr + vLElems * 3 + loopA * dimR);
                // L1
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                // L2
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < fold3; i++) {
            mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + vLElems + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg2, srcAddr + vLElems * 2 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg3, srcAddr + vLElems * 3 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg4, srcAddr + vLElems * 4 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg5, srcAddr + vLElems * 5 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg6, srcAddr + vLElems * 6 + loopA * dimR);
                AscendC::MicroAPI::DataCopy(vreg7, srcAddr + vLElems * 7 + loopA * dimR);
                // L1
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, mask);
                AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, mask);
                AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, mask);
                // L2
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                // L3
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR, vreg0, mask);
            }
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

        mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(maskElems);
        for (uint16_t i = 0; i < isBlkRepeat; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BLK>(vreg0, srcAddr + loopA * dimR);
                for (uint16_t loopR = 0; loopR < blkRepeats; loopR++) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BLK>(
                        vreg1, srcAddr + loopA * dimR + blkElems + blkElems * loopR);
                    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                }
                AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR, vreg0, mask);
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }

    template <typename DIndex, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceAR(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA, int32_t dimR,
                                    int32_t mainR, int32_t outerR, int32_t innerR)
    {
        constexpr uint16_t vLElems = IsB64<PromteT>() ? VL_LEN / sizeof(float) : VL_LEN / sizeof(PromteT);
        constexpr uint16_t blkElems = IsB64<PromteT>() ? BLOCK_SIZE / sizeof(float) : BLOCK_SIZE / sizeof(PromteT);
        if (outerR == 1 && innerR <= static_cast<int32_t>(blkElems * CONST2) &&
            dimA >= static_cast<int32_t>(vLElems / CONST2)) {
            if constexpr (IsSameType<PromteT, int32_t>::value || IsSameType<PromteT, uint32_t>::value ||
                          IsSameType<PromteT, float>::value) {
                return ReduceARLessBlock<DIndex, float, Trait>(dstAddr, srcAddr, dimA, dimR, innerR);
            } else {
                return ReduceARLessBlock<DIndex, half, Trait>(dstAddr, srcAddr, dimA, dimR, innerR);
            }
        }
        const uint16_t tailR = static_cast<uint16_t>(dimR - mainR);
        const uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
        const uint16_t inplaceRepeats = (tailR + vLElems - 1) / vLElems;

        // Calc the count of total-folds
        const uint16_t base = static_cast<uint16_t>(mainR / vLElems);
        const uint16_t folds = CalcFolds(base);

        // Calc main/tail folds
        const uint16_t avgFolds = BASE_FOLD; // Set main folds do 4 times
        const uint16_t mainTimes = folds / BASE_FOLD;
        const uint16_t tailFolds = folds % BASE_FOLD;
        const uint16_t fold1 = tailFolds == FOLD1 ? 1 : 0;
        const uint16_t fold2 = tailFolds == FOLD2 ? 1 : 0;
        const uint16_t fold3 = tailFolds == FOLD3 ? 1 : 0;
        const uint16_t rNum = mainR < vLElems ? mainR : vLElems;
        const int16_t blkRepeats = rNum / blkElems - 1;
        const uint16_t isBlkRepeat = blkRepeats > 0 ? 1 : 0;
        const uint16_t reservedR = rNum < blkElems ? rNum : blkElems;
        const uint16_t loopANum = dimA;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregMain;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregTail;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < needInplaceAdd; i++) {
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    uint32_t sreg0 = tailR;
                    for (uint16_t loopR = 0; loopR < inplaceRepeats; loopR++) {
                        mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(sreg0);
                        AscendC::MicroAPI::DataCopy(vregMain, srcAddr + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vregTail, srcAddr + mainR + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::Mul(vregMain, vregMain, vregTail, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR + loopR * vLElems, vregMain, mask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            // MainFolds need 16 register
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg2;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg3;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg4;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg5;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg6;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg7;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg8;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg9;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg10;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg11;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg12;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg13;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg14;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg15;
            AscendC::MicroAPI::MaskReg fullMask;
            AscendC::MicroAPI::UnalignReg uDst;
            AscendC::MicroAPI::UnalignReg uSrc;

            // Procsee main folds
            uint16_t loopRNum = base;
            fullMask = AscendC::MicroAPI::CreateMask<PromteT, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
                loopRNum = loopRNum >> avgFolds; // Calc the count of mainFolds
                uint16_t offsetR = loopRNum * vLElems;
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                        // L0
                        AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg1, srcAddr + offsetR + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg2, srcAddr + offsetR * 2 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg3, srcAddr + offsetR * 3 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg4, srcAddr + offsetR * 4 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg5, srcAddr + offsetR * 5 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg6, srcAddr + offsetR * 6 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg7, srcAddr + offsetR * 7 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg8, srcAddr + offsetR * 8 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg9, srcAddr + offsetR * 9 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg10, srcAddr + offsetR * 10 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg11, srcAddr + offsetR * 11 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg12, srcAddr + offsetR * 12 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg13, srcAddr + offsetR * 13 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg14, srcAddr + offsetR * 14 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg15, srcAddr + offsetR * 15 + loopA * dimR + loopR * vLElems);
                        // L1
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg8, fullMask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg9, fullMask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg10, fullMask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg11, fullMask);
                        AscendC::MicroAPI::Mul(vreg4, vreg4, vreg12, fullMask);
                        AscendC::MicroAPI::Mul(vreg5, vreg5, vreg13, fullMask);
                        AscendC::MicroAPI::Mul(vreg6, vreg6, vreg14, fullMask);
                        AscendC::MicroAPI::Mul(vreg7, vreg7, vreg15, fullMask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, fullMask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, fullMask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, fullMask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, fullMask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, fullMask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, fullMask);
                        // L3
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, fullMask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR + loopR * vLElems, vreg0, fullMask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            ReduceARFoldBase<PromteT, Trait>(dstAddr, srcAddr, dimR, blkElems, blkElems, vLElems, loopANum, fold1,
                                             fold2, fold3, isBlkRepeat, blkRepeats, mask, vreg0, vreg1, vreg2, vreg3,
                                             vreg4, vreg5, vreg6, vreg7);

            // Reduce to 1
            uint32_t sreg1 = 1U;
            mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(sreg1);
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::Duplicate(vreg0, 1);
                for (uint16_t loopR = 0; loopR < reservedR; loopR++) {
                    if constexpr (sizeof(PromteT) == CONST2) {
                        AscendC::MicroAPI::DataCopy<PromteT, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(
                            vreg1, srcAddr + loopA * dimR + loopR);
                    } else {
                        AscendC::MicroAPI::DataCopy<PromteT, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                            vreg1, srcAddr + loopA * dimR + loopR);
                    }
                    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                }
                AscendC::MicroAPI::DataCopyUnAlign((__ubuf__ PromteT*&)dstAddr, vreg0, uDst, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__ubuf__ PromteT*&)dstAddr, uDst, 0);
        }
    }

    template <typename DIndex, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceARI64(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA, int32_t dimR,
                                       int32_t mainR, int32_t outerR, int32_t innerR)
    {
        constexpr uint16_t vLElems = IsB64<PromteT>() ? VL_LEN / sizeof(float) : VL_LEN / sizeof(PromteT);
        constexpr uint16_t blkElems = IsB64<PromteT>() ? BLOCK_SIZE / sizeof(float) : BLOCK_SIZE / sizeof(PromteT);
        if (outerR == 1 && innerR <= static_cast<int32_t>(blkElems * CONST2) &&
            dimA >= static_cast<int32_t>(vLElems / CONST2)) {
            return ReduceARLessBlockI64<DIndex, float, Trait>(dstAddr, srcAddr, dimA, dimR, innerR);
        }
        const uint16_t tailR = dimR - mainR;
        const uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
        const uint16_t inplaceRepeats = (tailR + vLElems - 1) / vLElems;

        // Calc the count of total-folds
        const uint16_t base = mainR / vLElems;
        const uint16_t folds = CalcFolds(base);

        // Calc main/tail folds
        const uint16_t avgFolds = BASE_FOLD_I64; // Set main folds do 4 times
        const uint16_t mainTimes = folds / BASE_FOLD_I64;
        const uint16_t tailFolds = folds % BASE_FOLD_I64;
        const uint16_t fold1 = tailFolds == FOLD1 ? 1 : 0;
        const uint16_t fold2 = tailFolds == FOLD2 ? 1 : 0;
        const uint16_t rNum = mainR < vLElems ? mainR : vLElems;
        const int16_t blkRepeats = rNum / blkElems - 1;
        const uint16_t isBlkRepeat = blkRepeats > 0 ? 1 : 0;
        const uint16_t reservedR = rNum < blkElems ? rNum : blkElems;
        const uint16_t loopANum = dimA;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregMain;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregTail;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < needInplaceAdd; i++) {
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    uint32_t sreg0 = tailR;
                    for (uint16_t loopR = 0; loopR < inplaceRepeats; loopR++) {
                        mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(sreg0);
                        AscendC::MicroAPI::DataCopy(vregMain, srcAddr + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vregTail, srcAddr + mainR + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::Mul(vregMain, vregMain, vregTail, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR + loopR * vLElems, vregMain, mask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            // MainFolds need 16 register
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg2;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg3;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg4;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg5;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg6;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg7;
            AscendC::MicroAPI::MaskReg fullMask;
            AscendC::MicroAPI::UnalignReg uDst;
            AscendC::MicroAPI::UnalignReg uSrc;

            // Procsee main folds
            uint16_t loopRNum = base;
            fullMask = AscendC::MicroAPI::CreateMask<PromteT, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
                loopRNum = loopRNum >> avgFolds; // Calc the count of mainFolds
                uint16_t offsetR = loopRNum * vLElems;
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                        // L0
                        AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg1, srcAddr + offsetR + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg2, srcAddr + offsetR * 2 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg3, srcAddr + offsetR * 3 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg4, srcAddr + offsetR * 4 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg5, srcAddr + offsetR * 5 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg6, srcAddr + offsetR * 6 + loopA * dimR + loopR * vLElems);
                        AscendC::MicroAPI::DataCopy(vreg7, srcAddr + offsetR * 7 + loopA * dimR + loopR * vLElems);
                        // L1
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, fullMask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, fullMask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, fullMask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, fullMask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, fullMask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, fullMask);
                        // L3
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, fullMask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * dimR + loopR * vLElems, vreg0, fullMask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            ReduceARFoldBase<PromteT, Trait>(dstAddr, srcAddr, dimR, blkElems, blkElems, vLElems, loopANum, fold1,
                                             fold2, 0, isBlkRepeat, blkRepeats, mask, vreg0, vreg1, vreg2, vreg3, vreg4,
                                             vreg5, vreg6, vreg7);

            // Reduce to 1
            uint32_t sreg1 = 1U;
            mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(sreg1);
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                AscendC::MicroAPI::Duplicate(vreg0, 1);
                for (uint16_t loopR = 0; loopR < reservedR; loopR++) {
                    auto src = srcAddr + loopA * dimR + loopR;
                    AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, src);
                    AscendC::MicroAPI::DataCopyUnAlign(vreg1, uSrc, src, 1);
                    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                }
                AscendC::MicroAPI::DataCopyUnAlign(dstAddr, vreg0, uDst, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(dstAddr, uDst, 0);
        }
    }

    template <typename T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRAFoldBase(
        __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const int32_t dimA, uint32_t processA, const uint16_t vLElems,
        const uint16_t loopANum, uint16_t fold0, uint16_t fold1, uint16_t fold2, uint16_t fold3,
        AscendC::MicroAPI::MaskReg& mask, AscendC::MicroAPI::RegTensor<T, Trait>& vreg0,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg1, AscendC::MicroAPI::RegTensor<T, Trait>& vreg2,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg3, AscendC::MicroAPI::RegTensor<T, Trait>& vreg4,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg5, AscendC::MicroAPI::RegTensor<T, Trait>& vreg6,
        AscendC::MicroAPI::RegTensor<T, Trait>& vreg7)
    {
        for (uint16_t i = 0; i < fold0; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(dstAddr + loopA * vLElems, vreg0, mask);
            }
        }
        for (uint16_t i = 0; i < fold1; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
                // L0
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + dimA + loopA * vLElems);
                // L1
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(dstAddr + loopA * vLElems, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < fold2; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
                // L0
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + dimA + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg2, srcAddr + dimA * 2 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg3, srcAddr + dimA * 3 + loopA * vLElems);
                // L1
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                // L2
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(dstAddr + loopA * vLElems, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < fold3; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
                // L0
                AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg1, srcAddr + dimA + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg2, srcAddr + dimA * 2 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg3, srcAddr + dimA * 3 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg4, srcAddr + dimA * 4 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg5, srcAddr + dimA * 5 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg6, srcAddr + dimA * 6 + loopA * vLElems);
                AscendC::MicroAPI::DataCopy(vreg7, srcAddr + dimA * 7 + loopA * vLElems);
                // L1
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, mask);
                AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, mask);
                AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, mask);
                // L2
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                // L3
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(dstAddr + loopA * vLElems, vreg0, mask);
            }
        }
    }

    template <const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRA(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA, int32_t dimR,
                                    int32_t mainR)
    {
        constexpr uint16_t vLElems = VL_LEN / sizeof(PromteT);
        constexpr uint32_t dtypeSize = sizeof(PromteT);
        const uint16_t tailR = dimR - mainR;
        const uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;

        // Calc the count of total-folds
        const uint16_t folds = CalcFolds(mainR);

        // Calc main/tail folds
        const uint16_t avgFolds = BASE_FOLD; // Set main folds do 4 times
        const uint16_t mainTimes = folds / BASE_FOLD;
        const uint16_t tailFolds = folds % BASE_FOLD;
        const uint16_t fold0 = (tailFolds == 0) ? 1 : 0;
        const uint16_t fold1 = tailFolds == FOLD1 ? 1 : 0;
        const uint16_t fold2 = tailFolds == FOLD2 ? 1 : 0;
        const uint16_t fold3 = tailFolds == FOLD3 ? 1 : 0;

        // Process vLElems axisA each time
        const uint16_t loopANum = (dimA + vLElems - 1) / vLElems;
        uint32_t inplaceA = dimA;
        uint32_t processA = dimA;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregMain;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregTail;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < needInplaceAdd; i++) {
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(inplaceA);
                    for (uint16_t loopR = 0; loopR < tailR; loopR++) {
                        AscendC::MicroAPI::DataCopy(vregMain, srcAddr + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vregTail, srcAddr + mainR * dimA + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::Mul(vregMain, vregMain, vregTail, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * vLElems + loopR * dimA, vregMain, mask);
                    }
                }
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();

            // MainFolds need 16 register
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg2;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg3;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg4;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg5;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg6;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg7;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg8;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg9;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg10;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg11;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg12;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg13;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg14;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg15;

            // Procsee main folds
            uint16_t loopRNum = mainR;
            for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
                loopRNum = loopRNum >> avgFolds; // Calc the count of mainFolds
                uint16_t offsetR = loopRNum * dimA;
                uint32_t mainA = dimA;
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(mainA);
                    for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                        // L0
                        AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg1, srcAddr + offsetR + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg2, srcAddr + offsetR * 2 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg3, srcAddr + offsetR * 3 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg4, srcAddr + offsetR * 4 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg5, srcAddr + offsetR * 5 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg6, srcAddr + offsetR * 6 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg7, srcAddr + offsetR * 7 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg8, srcAddr + offsetR * 8 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg9, srcAddr + offsetR * 9 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg10, srcAddr + offsetR * 10 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg11, srcAddr + offsetR * 11 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg12, srcAddr + offsetR * 12 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg13, srcAddr + offsetR * 13 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg14, srcAddr + offsetR * 14 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg15, srcAddr + offsetR * 15 + loopA * vLElems + loopR * dimA);
                        // L1
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg8, mask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg9, mask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg10, mask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg11, mask);
                        AscendC::MicroAPI::Mul(vreg4, vreg4, vreg12, mask);
                        AscendC::MicroAPI::Mul(vreg5, vreg5, vreg13, mask);
                        AscendC::MicroAPI::Mul(vreg6, vreg6, vreg14, mask);
                        AscendC::MicroAPI::Mul(vreg7, vreg7, vreg15, mask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, mask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, mask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, mask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, mask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                        // L3
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * vLElems + loopR * dimA, vreg0, mask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            // Procsee tail folds
            ReduceRAFoldBase<PromteT, Trait>(dstAddr, srcAddr, dimA, dimA, vLElems, loopANum, fold0, fold1, fold2,
                                             fold3, mask, vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7);
        }
    }

    template <const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRAI64(__ubuf__ PromteT* dstAddr, __ubuf__ PromteT* srcAddr, int32_t dimA, int32_t dimR,
                                       int32_t mainR)
    {
        constexpr uint16_t vLElems = VL_LEN / sizeof(float);
        const uint16_t tailR = dimR - mainR;
        const uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;

        // Calc the count of total-folds
        const uint16_t folds = CalcFolds(mainR);

        // Calc main/tail folds
        const uint16_t avgFolds = BASE_FOLD_I64; // Set main folds do 4 times
        const uint16_t mainTimes = folds / avgFolds;
        const uint16_t tailFolds = folds % avgFolds;
        const uint16_t fold0 = (tailFolds == 0) ? 1 : 0;
        const uint16_t fold1 = tailFolds == FOLD1 ? 1 : 0;
        const uint16_t fold2 = tailFolds == FOLD2 ? 1 : 0;

        // Process vLElems axisA each time
        const uint16_t loopANum = (dimA + vLElems - 1) / vLElems;
        uint32_t inplaceA = dimA;
        uint32_t processA = dimA;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregMain;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vregTail;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < needInplaceAdd; i++) {
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(inplaceA);
                    for (uint16_t loopR = 0; loopR < tailR; loopR++) {
                        AscendC::MicroAPI::DataCopy(vregMain, srcAddr + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vregTail, srcAddr + mainR * dimA + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::Mul(vregMain, vregMain, vregTail, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * vLElems + loopR * dimA, vregMain, mask);
                    }
                }
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();

            // MainFolds need 16 register
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg0;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg1;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg2;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg3;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg4;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg5;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg6;
            AscendC::MicroAPI::RegTensor<PromteT, Trait> vreg7;

            // Procsee main folds
            uint16_t loopRNum = mainR;
            for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
                loopRNum = loopRNum >> avgFolds; // Calc the count of mainFolds
                uint16_t offsetR = loopRNum * dimA;
                uint32_t mainA = dimA;
                for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                    mask = AscendC::MicroAPI::UpdateMask<PromteT, Trait>(mainA);
                    for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                        // L0
                        AscendC::MicroAPI::DataCopy(vreg0, srcAddr + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg1, srcAddr + offsetR + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg2, srcAddr + offsetR * 2 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg3, srcAddr + offsetR * 3 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg4, srcAddr + offsetR * 4 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg5, srcAddr + offsetR * 5 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg6, srcAddr + offsetR * 6 + loopA * vLElems + loopR * dimA);
                        AscendC::MicroAPI::DataCopy(vreg7, srcAddr + offsetR * 7 + loopA * vLElems + loopR * dimA);
                        // L1
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg4, mask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg5, mask);
                        AscendC::MicroAPI::Mul(vreg2, vreg2, vreg6, mask);
                        AscendC::MicroAPI::Mul(vreg3, vreg3, vreg7, mask);
                        // L2
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg2, mask);
                        AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
                        // L3
                        AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                        AscendC::MicroAPI::DataCopy(srcAddr + loopA * vLElems + loopR * dimA, vreg0, mask);
                    }
                }
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            // Procsee tail folds
            ReduceRAFoldBase<PromteT, Trait>(dstAddr, srcAddr, dimA, dimA, vLElems, loopANum, fold0, fold1, fold2, 0,
                                             mask, vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7);
        }
    }

    template <class Pattern, class S>
    __aicore__ inline void UpdateCache(const LocalTensor<PromteT>& ubDst, const LocalTensor<PromteT>& ubSrc,
                                       int64_t index, S& shape)
    {
        DoCaching<Pattern, PromteT, S, REDUCE_OP_PROD>(ubDst, ubSrc, index, shape);
    }

    __aicore__ inline void BisectionPreHandle(const LocalTensor<PromteT>& dst, const LocalTensor<PromteT>& src0,
                                              const LocalTensor<PromteT>& src1, const int32_t& calCount)
    {
        AscendC::Mul(dst, src0, src1, calCount);
    }

    template <class Pattern, typename InputT, class S, class P>
    __aicore__ inline void PadValue(const LocalTensor<PromteT>& dst, S& shape, P& padding)
    {
        constexpr uint16_t vLElems = IsB64<PromteT>() ? VL_LEN / sizeof(float) : VL_LEN / sizeof(PromteT);
        constexpr uint16_t blkElems = IsB64<PromteT>() ? BLOCK_SIZE / sizeof(float) : BLOCK_SIZE / sizeof(PromteT);
        if constexpr (!Pattern::TailA) {
            if (padding.burstPaddingRepeat == 1U && shape.innerR <= blkElems * CONST2 &&
                shape.value[0] >= vLElems / CONST2) {
                return;
            }
        }
        PromteT padValue = GetPaddingValue<PromteT>();
        if constexpr (IsB64<PromteT>()) {
            DoPadding<AscendC::MicroAPI::RegTraitNumTwo, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        } else {
            DoPadding<AscendC::MicroAPI::RegTraitNumOne, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        }
    }

    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        U paddingValue = 1;
        return paddingValue;
    }
#endif
};

template <typename PromteT>
class ReduceSumOp : public ReduceOp<PromteT> {
public:
    __aicore__ inline ReduceSumOp() {}
#ifdef __CCE_AICORE__
    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
        AscendC::ReduceSum<PromteT, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);
    }

    template <class Pattern, bool isBatchInvariant,
              typename IsSameV<Pattern, __reducePattern::RA>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<PromteT>& dst,
                                   const LocalTensor<PromteT>& src)
    {
        if constexpr (isBatchInvariant && !IsB64<PromteT>()) {
            ReduceRAImpl<PromteT, AscendC::MicroAPI::RegTraitNumOne>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                     (__ubuf__ PromteT*)src.GetPhyAddr(),
                                                                     shape.value[1], shape.value[0]);
        } else {
            uint32_t srcShape[2] = {static_cast<uint32_t>(shape.value[0]), static_cast<uint32_t>(shape.value[1])};
            AscendC::ReduceSum<PromteT, AscendC::Pattern::Reduce::RA, true>(dst, src, srcShape, false);
        }
    }

    template <class Pattern, class S>
    __aicore__ inline void UpdateCache(const LocalTensor<PromteT>& ubDst, const LocalTensor<PromteT>& ubSrc,
                                       int64_t index, S& shape)
    {
        DoCaching<Pattern, PromteT, S, REDUCE_OP_SUM>(ubDst, ubSrc, index, shape);
    }

    __aicore__ inline void BisectionPreHandle(const LocalTensor<PromteT>& dst, const LocalTensor<PromteT>& src0,
                                              const LocalTensor<PromteT>& src1, const int32_t& calCount)
    {
        AscendC::Add(dst, src0, src1, calCount);
    }

    template <class Pattern, typename InputT, class S, class P>
    __aicore__ inline void PadValue(const LocalTensor<PromteT>& dst, S& shape, P& padding)
    {
        PromteT padValue = GetPaddingValue<PromteT>();
        if constexpr (IsB64<PromteT>()) {
            DoPadding<AscendC::MicroAPI::RegTraitNumTwo, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        } else {
            DoPadding<AscendC::MicroAPI::RegTraitNumOne, true, Pattern, InputT, PromteT>(
                (__ubuf__ PromteT*)dst.GetPhyAddr(), padValue, shape, padding);
        }
    }

    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        U paddingValue = 0;
        return paddingValue;
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRAOverVLVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t dimA,
                                                        uint32_t dimR, uint32_t mainR, uint32_t tailR,
                                                        uint16_t loopANum, uint16_t loopANumFinal, uint16_t folds,
                                                        uint16_t avgFolds, uint16_t foldZero, uint16_t foldOne,
                                                        uint16_t foldTwo, uint16_t foldThree)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
        uint16_t noNeedCopy = mainR > 1 ? 0 : 1;
        uint16_t mainTimes = folds / avgFolds;
        // Process vlSize axisA each time
        uint32_t inplaceA = dimA;
        uint32_t processA = dimA;
        uint32_t tailA = dimA;
        uint32_t copyA = dimA;
        uint32_t dtypeSize = sizeof(T);
        uint32_t aTailOffset = mainR * dimA;

        __ubuf__ T* addr;
        addr = srcAddr;
        AscendC::MicroAPI::RegTensor<T, Trait> vregMain;
        AscendC::MicroAPI::RegTensor<T, Trait> vregTail;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < noNeedCopy; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(inplaceA);
                AscendC::MicroAPI::LoadAlign(vregMain, srcAddr + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vregTail, srcAddr + loopA * vlSize + aTailOffset);
                AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, mask);
                AscendC::MicroAPI::StoreAlign(dstAddr + loopA * vlSize, vregMain, mask);
            }
            return;
        }
        // Process mainR and tailR
        for (uint16_t i = 0; i < needInplaceAdd; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(inplaceA);
                for (uint16_t loopR = 0; loopR < static_cast<uint16_t>(tailR); loopR++) {
                    AscendC::MicroAPI::LoadAlign(vregMain, srcAddr + loopA * vlSize + loopR * dimA);
                    AscendC::MicroAPI::LoadAlign(vregTail, srcAddr + loopA * vlSize + aTailOffset + loopR * dimA);
                    AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, mask);
                    AscendC::MicroAPI::StoreAlign(addr + loopA * vlSize + loopR * dimA, vregMain, mask);
                }
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // MainFolds need 16 register
        AscendC::MicroAPI::RegTensor<T, Trait> vreg0;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg1;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg2;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg3;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg4;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg5;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg6;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg7;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg8;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg9;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg10;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg11;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg12;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg13;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg14;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg15;

        // Process main folds
        uint16_t loopRNum = mainR;
        for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
            loopRNum = loopRNum >> avgFolds;
            uint32_t mainA = dimA;
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(mainA);
                __ubuf__ T* tmpSrcAddr = addr + loopA * vlSize;
                for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                    // L0
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg4, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg5, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg8, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg9, tmpSrcAddr,
                                                                                                      dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg10,
                                                                                                      tmpSrcAddr, dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg11,
                                                                                                      tmpSrcAddr, dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg12,
                                                                                                      tmpSrcAddr, dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg13,
                                                                                                      tmpSrcAddr, dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg14,
                                                                                                      tmpSrcAddr, dimA);
                    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg15,
                                                                                                      tmpSrcAddr, dimA);
                    // L1
                    AscendC::MicroAPI::Add(vreg0, vreg0, vreg8, mask);
                    AscendC::MicroAPI::Add(vreg1, vreg1, vreg9, mask);
                    AscendC::MicroAPI::Add(vreg2, vreg2, vreg10, mask);
                    AscendC::MicroAPI::Add(vreg3, vreg3, vreg11, mask);
                    AscendC::MicroAPI::Add(vreg4, vreg4, vreg12, mask);
                    AscendC::MicroAPI::Add(vreg5, vreg5, vreg13, mask);
                    AscendC::MicroAPI::Add(vreg6, vreg6, vreg14, mask);
                    AscendC::MicroAPI::Add(vreg7, vreg7, vreg15, mask);
                    // L2
                    AscendC::MicroAPI::Add(vreg0, vreg0, vreg4, mask);
                    AscendC::MicroAPI::Add(vreg1, vreg1, vreg5, mask);
                    AscendC::MicroAPI::Add(vreg2, vreg2, vreg6, mask);
                    AscendC::MicroAPI::Add(vreg3, vreg3, vreg7, mask);
                    // L3
                    AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
                    AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
                    // L4
                    AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
                    AscendC::MicroAPI::StoreAlign(addr + loopA * vlSize + loopR * dimA, vreg0, mask);
                }
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        for (uint16_t i = 0; i < foldOne; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(tailA);
                // L0
                AscendC::MicroAPI::LoadAlign(vreg0, addr + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
                // L1
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(tailA);
                // L0
                AscendC::MicroAPI::LoadAlign(vreg0, addr + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
                // L1
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
                // L2
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }

        for (uint16_t i = 0; i < foldThree; i++) {
            for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(tailA);
                // L0
                AscendC::MicroAPI::LoadAlign(vreg0, addr + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg4, addr + 4 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg5, addr + 5 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg6, addr + 6 * dimA + loopA * vlSize);
                AscendC::MicroAPI::LoadAlign(vreg7, addr + 7 * dimA + loopA * vlSize);
                // L1
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg4, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg5, mask);
                AscendC::MicroAPI::Add(vreg2, vreg2, vreg6, mask);
                AscendC::MicroAPI::Add(vreg3, vreg3, vreg7, mask);
                // L2
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
                // L3
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }

        // Reduce to 1
        for (uint16_t i = 0; i < foldZero; i++) {
            for (uint16_t loopA = 0; loopA < loopANumFinal; loopA++) {
                mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
                AscendC::MicroAPI::LoadAlign(vreg0, addr + loopA * vlSize);
                AscendC::MicroAPI::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
            }
        }
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRAOverVLImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t dimA, uint32_t dimR)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint32_t mainR = MainR(dimR, false, vlSize);
        uint32_t tailR = dimR - mainR;

        uint16_t loopANum = Ops::Base::CeilDiv(dimA, vlSize);
        // move by fold zero only if R axis is 1
        uint16_t loopANumFinal = loopANum;
        if (mainR == 1) {
            ReduceCopyOutImpl<T>(dstAddr, srcAddr, dimA);
            return;
        }

        uint16_t folds = CalcFolds(mainR);
        uint16_t avgFolds = BASE_FOLD;
        uint16_t tailFolds = folds % avgFolds;
        uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
        uint16_t foldOne = (tailFolds == FOLD1) ? 1 : 0;
        uint16_t foldTwo = (tailFolds == FOLD2) ? 1 : 0;
        uint16_t foldThree = (tailFolds == FOLD3) ? 1 : 0;

        ReduceRAOverVLVFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR, mainR, tailR, loopANum, loopANumFinal, folds,
                                       avgFolds, foldZero, foldOne, foldTwo, foldThree);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRALessThanVLDimR1VFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr,
                                                                 uint32_t dimA, uint32_t dimR)
    {
        AscendC::MicroAPI::RegTensor<T, Trait> vregTmp;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<T, Trait>(dimA);
        AscendC::MicroAPI::LoadAlign(vregTmp, srcAddr);
        AscendC::MicroAPI::StoreAlign(dstAddr, vregTmp, mask);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRALessThanVLVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA,
                                                            uint32_t dimR, uint32_t mainR, uint32_t tailR,
                                                            uint16_t folds, uint16_t avgFolds, uint16_t foldZero,
                                                            uint16_t foldOne, uint16_t foldTwo, uint16_t foldThree)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
        uint16_t mainTimes = folds / avgFolds;
        // Process vlSize axisA each time
        uint32_t processA = dimA;
        uint32_t dtypeSize = sizeof(T);
        uint32_t aTailOffset = mainR * dimA;
        uint32_t copyNum = (mainR - tailR) * dimA;
        uint32_t tailNum = tailR * dimA;
        uint16_t loopRNum = mainR;

        __ubuf__ T* addr;
        AscendC::MicroAPI::MaskReg mask;
        mask = AscendC::MicroAPI::UpdateMask<T, Trait>(processA);
        AscendC::MicroAPI::MaskReg counterMask;
        addr = srcAddr;

        AscendC::MicroAPI::RegTensor<T, Trait> vregMain;
        AscendC::MicroAPI::RegTensor<T, Trait> vregTail;
        // Process mainR and tailR
        for (uint16_t i = 0; i < needInplaceAdd; i++) {
            uint16_t tailRepeat = CeilDivision(tailNum, vlSize);
            for (uint16_t loopTail = 0; loopTail < tailRepeat; loopTail++) {
                counterMask = AscendC::MicroAPI::UpdateMask<T, Trait>(tailNum);
                AscendC::MicroAPI::LoadAlign(vregMain, srcAddr + loopTail * vlSize);
                AscendC::MicroAPI::LoadAlign(vregTail, srcAddr + aTailOffset + loopTail * vlSize);
                AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, counterMask);
                AscendC::MicroAPI::StoreAlign(addr + loopTail * vlSize, vregMain, counterMask);
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // MainFolds need 16 register
        AscendC::MicroAPI::RegTensor<T, Trait> vreg0;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg1;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg2;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg3;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg4;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg5;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg6;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg7;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg8;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg9;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg10;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg11;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg12;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg13;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg14;
        AscendC::MicroAPI::RegTensor<T, Trait> vreg15;

        // Process main folds
        for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
            loopRNum = loopRNum >> avgFolds;
            auto tmpSrcAddr = addr;
            for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
                // L0
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg4, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg5, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg8, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg9, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg10, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg11, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg12, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg13, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg14, tmpSrcAddr,
                                                                                                  dimA);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg15, tmpSrcAddr,
                                                                                                  dimA);
                // L1
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg8, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg9, mask);
                AscendC::MicroAPI::Add(vreg2, vreg2, vreg10, mask);
                AscendC::MicroAPI::Add(vreg3, vreg3, vreg11, mask);
                AscendC::MicroAPI::Add(vreg4, vreg4, vreg12, mask);
                AscendC::MicroAPI::Add(vreg5, vreg5, vreg13, mask);
                AscendC::MicroAPI::Add(vreg6, vreg6, vreg14, mask);
                AscendC::MicroAPI::Add(vreg7, vreg7, vreg15, mask);
                // L2
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg4, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg5, mask);
                AscendC::MicroAPI::Add(vreg2, vreg2, vreg6, mask);
                AscendC::MicroAPI::Add(vreg3, vreg3, vreg7, mask);
                // L3
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
                AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
                // L4
                AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::StoreAlign(addr + loopR * dimA, vreg0, mask);
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // Process tail folds
        for (uint16_t i = 0; i < foldOne; i++) {
            // L0
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
            // L1
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr, vreg0, mask);
        }

        for (uint16_t i = 0; i < foldTwo; i++) {
            // L0
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, addr, dimA);
            // L1
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
            AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
            // L2
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr, vreg0, mask);
        }

        for (uint16_t i = 0; i < foldThree; i++) {
            // L0
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg3, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg4, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg5, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, addr, dimA);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, addr, dimA);
            // L1
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg4, mask);
            AscendC::MicroAPI::Add(vreg1, vreg1, vreg5, mask);
            AscendC::MicroAPI::Add(vreg2, vreg2, vreg6, mask);
            AscendC::MicroAPI::Add(vreg3, vreg3, vreg7, mask);
            // L2
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg2, mask);
            AscendC::MicroAPI::Add(vreg1, vreg1, vreg3, mask);
            // L3
            AscendC::MicroAPI::Add(vreg0, vreg0, vreg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr, vreg0, mask);
        }

        // Reduce to 1
        for (uint16_t i = 0; i < foldZero; i++) {
            AscendC::MicroAPI::LoadAlign(vreg0, addr);
            AscendC::MicroAPI::StoreAlign(dstAddr, vreg0, mask);
        }
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRALessThanVLImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA,
                                                  uint32_t dimR)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint32_t mainR = MainR(dimR, false, vlSize);
        uint32_t tailR = dimR - mainR;
        if (dimR == 1) {
            ReduceRALessThanVLDimR1VFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR);
            return;
        }

        uint16_t folds = CalcFolds(mainR);
        uint16_t avgFolds = BASE_FOLD;
        uint16_t tailFolds = folds % avgFolds;
        uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
        uint16_t foldOne = (tailFolds == FOLD1) ? 1 : 0;
        uint16_t foldTwo = (tailFolds == FOLD2) ? 1 : 0;
        uint16_t foldThree = (tailFolds == FOLD3) ? 1 : 0;

        ReduceRALessThanVLVFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR, mainR, tailR, folds, avgFolds, foldZero,
                                           foldOne, foldTwo, foldThree);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRAConcatDimR1VFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA,
                                                             uint32_t dimR)
    {
        AscendC::MicroAPI::RegTensor<T, Trait> vregTmp;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<T, Trait>(dimA);
        AscendC::MicroAPI::LoadAlign(vregTmp, srcAddr);
        AscendC::MicroAPI::StoreAlign(dstAddr, vregTmp, mask);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRAConcatDimR2VFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA,
                                                             uint32_t dimR)
    {
        AscendC::MicroAPI::RegTensor<T, Trait> vregMain;
        AscendC::MicroAPI::RegTensor<T, Trait> vregTail;
        uint32_t maskScalar = dimA;
        AscendC::MicroAPI::MaskReg counterMask = AscendC::MicroAPI::UpdateMask<T, Trait>(maskScalar);
        AscendC::MicroAPI::LoadAlign(vregMain, srcAddr);
        AscendC::MicroAPI::LoadAlign(vregTail, srcAddr + dimA);
        AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, counterMask);
        AscendC::MicroAPI::StoreAlign(dstAddr, vregMain, counterMask);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    static __simd_vf__ inline void ReduceRAConcatVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA,
                                                        uint32_t dimR, uint16_t foldTime, uint32_t mainR,
                                                        uint32_t tailR)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint16_t needInplaceAdd = tailR > 0 ? 1 : 0;
        // Process vlSize axisA each time
        uint32_t processA = dimA;
        uint32_t dtypeSize = sizeof(T);
        uint32_t aTailOffset = mainR * dimA;
        // do mainR-tailR copy, do tailR binary op
        uint32_t copyNum = (mainR - tailR) * dimA;
        uint32_t tailNum = tailR * dimA;
        uint16_t loopDataNum = mainR * dimA;

        __ubuf__ T* addr;
        AscendC::MicroAPI::MaskReg fullMask;
        fullMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        AscendC::MicroAPI::MaskReg counterMask;
        addr = srcAddr;

        AscendC::MicroAPI::RegTensor<T, Trait> vregMain;
        AscendC::MicroAPI::RegTensor<T, Trait> vregTail;
        // Process mainR and tailR
        for (uint16_t i = 0; i < needInplaceAdd; i++) {
            uint16_t tailRepeat = CeilDivision(tailNum, vlSize);
            for (uint16_t loopTail = 0; loopTail < tailRepeat; loopTail++) {
                counterMask = AscendC::MicroAPI::UpdateMask<T, Trait>(tailNum);
                AscendC::MicroAPI::LoadAlign(vregMain, srcAddr + loopTail * vlSize);
                AscendC::MicroAPI::LoadAlign(vregTail, srcAddr + aTailOffset + loopTail * vlSize);
                AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, counterMask);
                AscendC::MicroAPI::StoreAlign(addr + loopTail * vlSize, vregMain, counterMask);
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        for (uint16_t fold = 0; fold < foldTime; fold++) {
            mainR = mainR >> 1;
            uint32_t foldDataNum = mainR * dimA;
            uint16_t foldRepeat = CeilDivision(foldDataNum, vlSize);
            for (uint16_t i = 0; i < foldRepeat; i++) {
                AscendC::MicroAPI::LoadAlign(vregMain, addr + i * vlSize);
                AscendC::MicroAPI::LoadAlign(vregTail, addr + foldDataNum + i * vlSize);
                AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, fullMask);
                AscendC::MicroAPI::StoreAlign(addr + i * vlSize, vregMain, fullMask);
            }
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // final fold is less than vl, no repeat
        uint32_t maskScalar = dimA;
        counterMask = AscendC::MicroAPI::UpdateMask<T, Trait>(maskScalar);
        AscendC::MicroAPI::LoadAlign(vregMain, addr);
        AscendC::MicroAPI::LoadAlign(vregTail, addr + dimA);
        AscendC::MicroAPI::Add(vregMain, vregMain, vregTail, counterMask);
        AscendC::MicroAPI::StoreAlign(dstAddr, vregMain, counterMask);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRAConcatImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        uint16_t foldTime = FindClosestPowerOfTwo(dimR);
        uint32_t mainR = 1 << foldTime;
        // last fold not in main loop, main R == 1 will not enter main loop
        foldTime = foldTime - 1;
        uint32_t tailR = dimR - mainR;

        if (dimR == 1) {
            ReduceRAConcatDimR1VFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR);
            return;
        } else if (dimR == 2) {
            ReduceRAConcatDimR2VFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR);
            return;
        }

        ReduceRAConcatVFImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR, foldTime, mainR, tailR);
    }

    template <class T, const AscendC::MicroAPI::RegTrait& Trait>
    __aicore__ inline void ReduceRAImpl(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dimA, uint32_t dimR)
    {
        constexpr uint16_t vlSize = VL_LEN / sizeof(T);
        if (dimA <= vlSize / REGULAR_FOLD_NUM || dimA > U16_STRIDE) {
            ReduceRAConcatImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR);
        } else if (dimA <= vlSize) {
            ReduceRALessThanVLImpl<T, Trait>(dstAddr, srcAddr, dimA, dimR);
        } else {
            ReduceRAOverVLImpl<T, Trait>(dstAddr, srcAddr, static_cast<uint16_t>(dimA), dimR);
        }
    }

    template <class T>
    static __simd_vf__ inline void ReduceCopyOutImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint32_t calCount)
    {
        uint16_t repeatElm = VL_LEN;
        uint32_t sreg = calCount * sizeof(T);
        uint16_t repeatTime = CeilDivision(sreg, repeatElm);
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<uint8_t> srcReg;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = AscendC::MicroAPI::UpdateMask<uint8_t>(sreg);
            AscendC::MicroAPI::LoadAlign(srcReg, (__ubuf__ uint8_t*)src + i * repeatElm);
            AscendC::MicroAPI::StoreAlign((__ubuf__ uint8_t*)dst + i * repeatElm, srcReg, preg);
        }
    }

#endif
};

template <typename PromteT>
class ReduceAnyOp : public ReduceMaxOp<PromteT> {
public:
    __aicore__ inline ReduceAnyOp() {}
#ifdef __CCE_AICORE__
    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        return 0;
    }
#endif
};

template <typename PromteT>
class ReduceAllOp : public ReduceMinOp<PromteT> {
public:
    __aicore__ inline ReduceAllOp() {}
#ifdef __CCE_AICORE__
    template <typename U>
    __aicore__ inline U GetPaddingValue()
    {
        return 1;
    }
#endif
};
} // namespace Vec
} // namespace Base
} // namespace Ops
#endif
