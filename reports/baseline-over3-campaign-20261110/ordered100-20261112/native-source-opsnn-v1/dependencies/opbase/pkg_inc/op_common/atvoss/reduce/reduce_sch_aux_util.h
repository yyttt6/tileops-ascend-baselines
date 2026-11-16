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
 * \file reduce_sch_aux_util.h
 * \brief reduce schedule dispatch&impl
 */

#ifndef _REDUCE_SCH_AUX_UTIL_H_
#define _REDUCE_SCH_AUX_UTIL_H_
#include "reduce_util.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/sync.h"
#include "atvoss/util/buffer.h"

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {
constexpr static int64_t CONST_SIXTY_THREE = 63;
template <typename C, auto t>
struct Helper {};

template <typename, typename = void>
struct HasCopyIn {
    static constexpr bool value = false;
};

template <typename C>
struct HasCopyIn<C, decltype(void(
                        &C::template CopyIn<struct DummyPos, struct DummyPattern, struct DummyT, struct DummyV>))> {
    static constexpr bool value = true;
};

template <typename C>
__aicore__ inline constexpr static bool CheckInit(int32_t)
{
    return false;
}

template <typename C>
__aicore__ inline constexpr static bool CheckInit(Helper<C, &C::Init>*)
{
    return true;
}

template <typename C>
__aicore__ inline constexpr static bool CheckInitTiling(int32_t)
{
    return false;
}

template <typename C>
__aicore__ inline constexpr static bool CheckInitTiling(Helper<C, &C::InitTiling>*)
{
    return true;
}

template <typename C>
__aicore__ inline constexpr static bool CheckProcess(int32_t)
{
    return false;
}

template <typename C>
__aicore__ inline constexpr static bool CheckProcess(Helper<C, &C::Process>*)
{
    return true;
}

template <typename C>
__aicore__ inline constexpr static bool CheckCopyIn()
{
    return HasCopyIn<C>::value;
}

template <typename C>
__aicore__ inline constexpr static bool CheckCopyOut(int32_t)
{
    return false;
}

template <typename C>
__aicore__ inline constexpr static bool CheckCopyOut(Helper<C, &C::CopyOut>*)
{
    return true;
}

template <class ElemDag, class DataType, int32_t Size, bool IsVoid>
struct OutputDType {
    // 获取当前位置的输入dtype
    using ElemOp = typename ElemDag::FunList::template At<Size>;
    using T = typename ElemOp::template FunInArgType<0>;
};

template <class ElemDag, class DataType, int32_t Size>
struct OutputDType<ElemDag, DataType, Size, true> {
    using T = DataType;
};
template <class ElemDag, class DataType, bool IsStageOne, bool IsVoid>
struct InputDType {
    using ElemOp = typename ElemDag::FunList::template At<0>;
    using T = typename ElemOp::template FunRetArgType<0>;
};

template <class ElemDag, class DataType, bool IsStageOne>
struct InputDType<ElemDag, DataType, IsStageOne, true> {
    using T = DataType;
};

template <class ElemDag, class DataType>
struct InputDType<ElemDag, DataType, false, true> {
    using T = DataType;
};

template <class ElemDag, class DataType>
struct InputDType<ElemDag, DataType, false, false> {
    using T = DataType;
};

__aicore__ inline uint64_t FindNearestPower2(const uint64_t value)
{
    if (value == 0) {
        return 0;
    } else if (value <= CONST2) {
        return CONST1;
    } else if (value <= CONST4) {
        return CONST2;
    } else {
        const uint64_t num = value - CONST1;
        const uint64_t pow = CONST_SIXTY_THREE - ScalarCountLeadingZero(num);
        return (CONST1 << pow);
    }
}

__aicore__ inline uint64_t CalLog2(uint64_t value)
{
    uint64_t res = 0;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    return res;
}

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<CONST1>(idx ^ (idx + CONST1)) - CONST1;
}

template <HardEvent EVENT>
__aicore__ inline void SetEvent(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<EVENT>(eventId);
    WaitFlag<EVENT>(eventId);
}

template <auto LoopInfo>
__aicore__ inline constexpr bool IsBlockCutA()
{
    return !(LoopInfo->loopRCount > 0);
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopAxis(int32_t axis)
{
    if constexpr (LoopInfo->loopACount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopACount; i++) {
            if (LoopInfo->loopAAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopRCount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopRCount; i++) {
            if (LoopInfo->loopRAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopInnerACount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopInnerACount; i++) {
            if (LoopInfo->loopInnerAAxis[i] == axis) {
                return true;
            }
        }
    }

    if constexpr (LoopInfo->loopInnerRCount > 0) {
        for (int32_t i = 0; i < LoopInfo->loopInnerRCount; i++) {
            if (LoopInfo->loopInnerRAxis[i] == axis) {
                return true;
            }
        }
    }
    return false;
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopSpliteRAxis(int32_t axis)
{
    if constexpr (LoopInfo->loopInnerRCount) {
        if (LoopInfo->loopInnerRAxis[LoopInfo->loopInnerRCount - 1] == axis) {
            return true;
        }
    }
    if constexpr (LoopInfo->loopRCount) {
        if (LoopInfo->loopRAxis[LoopInfo->loopRCount - 1] == axis) {
            return true;
        }
    }
    return false;
}

template <auto LoopInfo>
__aicore__ inline bool IsLoopSpliteAAxis(int32_t axis)
{
    if constexpr (LoopInfo->loopACount) {
        if (LoopInfo->loopAAxis[LoopInfo->loopACount - 1] == axis) {
            return true;
        }
    }
    return false;
}

template <auto LoopInfo, int32_t DIM, class ADDR, class TILING>
__aicore__ inline constexpr uint64_t GetBurstLen(ADDR& iterAddr, TILING& tiling)
{
    return iterAddr[DIM].stride * tiling->stride[DIM];
}

template <auto LoopInfo, class ADDR, class TILING>
__aicore__ inline uint64_t GetRepeatStride(int32_t Dim, ADDR& iterAddr, TILING& tiling, uint64_t& stride)
{
    stride = tiling->stride[Dim];
    return iterAddr[Dim].stride;
}

template <bool FirstA>
__aicore__ inline bool IsAxisA(int32_t idx)
{
    if constexpr (FirstA) {
        return idx % CONST2 == 0;
    } else {
        return idx % CONST2 == 1;
    }
}

template <auto LoopInfo, bool TailA, int32_t DIM, class ADDR>
__aicore__ inline int32_t CaculateInnerA(ADDR& iterAddr)
{
    // 从后往前遍历到A的最内切分轴
    constexpr int32_t start = TailA ? DIM - 1 : DIM - 2;
    constexpr int32_t end = LoopInfo->loopACount > 0 ? LoopInfo->loopAAxis[LoopInfo->loopACount - 1] : -1;
    int32_t innerA = 1;
    for (int32_t idx = start; idx > end; idx -= 2) {
        innerA = innerA * iterAddr[idx].stride;
    }
    return innerA;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo0()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo1()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo2()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo3()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo4()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo5()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM8, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .innerPatternID = PatternID % CONST10};
    return schInfo;
}

__aicore__ inline constexpr ReduceSchLoopInfo GetGroupSchLoopInfo()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PATTERN_RA,
                                           .reduceDichotomy = CONST1,
                                           .loopACount = CONST1,
                                           .loopAAxis = {DIM1},
                                           .loopRCount = CONST0,
                                           .loopRAxis = {},
                                           .loopInnerACount = CONST0,
                                           .loopInnerAAxis = {},
                                           .loopInnerRCount = CONST1,
                                           .loopInnerRAxis = {DIM0},
                                           .innerPatternID = PATTERN_RA};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo()
{
    if constexpr (LoopARCount / CONST10 == CONST0) {
        return GetSchLoopInfo0<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST1) {
        return GetSchLoopInfo1<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST2) {
        return GetSchLoopInfo2<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST3) {
        return GetSchLoopInfo3<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST4) {
        return GetSchLoopInfo4<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST5) {
        return GetSchLoopInfo5<PatternID, LoopARCount, LoopInnerARCount>();
    }
}
} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops
#endif