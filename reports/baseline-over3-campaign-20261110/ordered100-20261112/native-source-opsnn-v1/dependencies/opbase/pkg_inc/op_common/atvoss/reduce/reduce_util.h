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
 * \file reduce_util.h
 * \brief reduce template struct
 */

#ifndef _REDUCE_UTIL_H_
#define _REDUCE_UTIL_H_
#include <stdint.h>
#ifdef __CCE_AICORE__
#include "atvoss/util/kernel_vec_intf_apt.h"
#endif

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM4 = 4;
constexpr int32_t DIM5 = 5;
constexpr int32_t DIM6 = 6;
constexpr int32_t DIM7 = 7;
constexpr int32_t DIM8 = 8;
constexpr int32_t DIM9 = 9;

constexpr int32_t CONST0 = 0;
constexpr int32_t CONST1 = 1;
constexpr int32_t CONST2 = 2;
constexpr int32_t CONST3 = 3;
constexpr int32_t CONST4 = 4;
constexpr int32_t CONST5 = 5;
constexpr int32_t CONST6 = 6;
constexpr int32_t CONST7 = 7;
constexpr int32_t CONST8 = 8;
constexpr int32_t CONST9 = 9;
constexpr int32_t CONST10 = 10;

constexpr int32_t MAX_DIM = 9;
constexpr int32_t MAX_LOOP_DIM = (MAX_DIM + 1) / 2;
constexpr int32_t PATTERN_A = 10;
constexpr int32_t PATTERN_RA = 0;
constexpr int32_t PATTERN_AR = 1;
constexpr int32_t PATTERN_ARA = 2;
constexpr int32_t PATTERN_ARAR = 3;
constexpr int32_t PATTERN_ARARA = 4;
constexpr int32_t PATTERN_ARARAR = 5;
constexpr int32_t PATTERN_ARARARA = 6;
constexpr int32_t PATTERN_ARARARAR = 7;
constexpr int32_t PATTERN_ARARARARA = 8;

template <class T, class U>
struct IsSameV {};
template <class T>
struct IsSameV<T, T> {
    using Type = T;
};

namespace __reducePattern {
template <uint32_t id, bool firstA, bool tailA, int32_t dim>
struct PatternConstInfo {
    constexpr static uint32_t ID = id;
    constexpr static bool FirstA = firstA;
    constexpr static bool TailA = tailA;
    constexpr static int32_t Dim = dim;
};

struct A : public PatternConstInfo<PATTERN_A, true, true, CONST1> {};
struct RA : public PatternConstInfo<PATTERN_RA, false, true, CONST2> {};
struct AR : public PatternConstInfo<PATTERN_AR, true, false, CONST2> {};
struct ARA : public PatternConstInfo<PATTERN_ARA, true, true, CONST3> {};
struct ARAR : public PatternConstInfo<PATTERN_ARAR, true, false, CONST4> {};
struct ARARA : public PatternConstInfo<PATTERN_ARARA, true, true, CONST5> {};
struct ARARAR : public PatternConstInfo<PATTERN_ARARAR, true, false, CONST6> {};
struct ARARARA : public PatternConstInfo<PATTERN_ARARARA, true, true, CONST7> {};
struct ARARARAR : public PatternConstInfo<PATTERN_ARARARAR, true, false, CONST8> {};
struct ARARARARA : public PatternConstInfo<PATTERN_ARARARARA, true, true, CONST9> {};

template <int32_t id>
struct GetPattern {};

template <>
struct GetPattern<PATTERN_A> {
    using T = A;
};

template <>
struct GetPattern<PATTERN_RA> {
    using T = RA;
};
template <>
struct GetPattern<PATTERN_AR> {
    using T = AR;
};
template <>
struct GetPattern<PATTERN_ARA> {
    using T = ARA;
};
template <>
struct GetPattern<PATTERN_ARAR> {
    using T = ARAR;
};
template <>
struct GetPattern<PATTERN_ARARA> {
    using T = ARARA;
};
template <>
struct GetPattern<PATTERN_ARARAR> {
    using T = ARARAR;
};
template <>
struct GetPattern<PATTERN_ARARARA> {
    using T = ARARARA;
};
template <>
struct GetPattern<PATTERN_ARARARAR> {
    using T = ARARARAR;
};
template <>
struct GetPattern<PATTERN_ARARARARA> {
    using T = ARARARARA;
};
} // namespace __reducePattern

#ifdef __CCE_AICORE__
const half HALF_MAX_VALUE = INFINITY;
const half HALF_MIN_VALUE = -INFINITY;
const bfloat16_t BFLOAT16_MAX_VALUE = INFINITY;
const bfloat16_t BFLOAT16_MIN_VALUE = -INFINITY;
const float FLOAT_MAX_VALUE = INFINITY;
const float FLOAT_MIN_VALUE = -INFINITY;
const uint8_t UINT8_MIN_VALUE = 0;
const uint8_t UINT8_MAX_VALUE = 255;
const int8_t INT8_MIN_VALUE = -128;
const int8_t INT8_MAX_VALUE = 127;

template <int32_t dim>
struct Shape {
    int64_t value[dim];
    int64_t innerR;
    int64_t outerR;
};

template <int32_t dim>
struct SliceView {
    uint64_t addr;
    uint32_t isBlockAligned; // burstLen的dstStride是否block对齐
    uint64_t axisSize;
    struct {
        uint64_t start = 0;
        uint64_t repeat = 1; // 当前轴的重复次数
        uint64_t srcStride = 0;
        uint64_t dstStride = 0;
        uint64_t idx = 0;
        bool isSliceNum = false;
        bool isAxisA = false;
    } axis[dim];
};

struct PaddingParam {
    int32_t burstPaddingStart = 0;  // burstLen 补pad的起始位置
    int32_t burstPaddingLen = 0;    // burstLen 补pad的长度
    int32_t burstPaddingRepeat = 0; // burstLen 补pad的循环次数
    int32_t rPaddingStart = 0;      // r轴 补pad的起始位置
    int32_t rPaddingLen = 0;        // r轴 补pad的长度
    int32_t rPaddingRepeat = 0;     // r轴 补pad的循环次数
    int32_t aPaddingStart = 0;      // a轴 补pad的起始位置
    int32_t aPaddingLen = 0;        // a轴 补pad的长度
    int32_t aPaddingRepeat = 0;     // a轴 补pad的循环次数
};

struct SliceFactors {
    uint64_t sliceShapeFactor;
    uint64_t sliceNumFactor;
};

namespace __reduceType {
template <typename DType>
struct GetPromoteType {};

template <>
struct GetPromoteType<half> {
    using T = float;
};

template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};

template <>
struct GetPromoteType<float> {
    using T = float;
};

template <>
struct GetPromoteType<int32_t> {
    using T = int32_t;
};

template <>
struct GetPromoteType<int64_t> {
    using T = int64_t;
};

template <>
struct GetPromoteType<int8_t> {
    using T = half;
};

template <>
struct GetPromoteType<uint8_t> {
    using T = half;
};

template <>
struct GetPromoteType<bool> {
    using T = half;
};
} // namespace __reduceType

struct ReduceSchLoopInfo {
    int32_t patternID;
    int32_t reduceDichotomy;
    int32_t loopACount;
    int32_t loopAAxis[MAX_LOOP_DIM];
    int32_t loopRCount;
    int32_t loopRAxis[MAX_DIM];

    int32_t loopInnerACount;
    int32_t loopInnerAAxis[MAX_DIM];
    int32_t loopInnerRCount;
    int32_t loopInnerRAxis[MAX_DIM];
    int32_t innerPatternID;
};

#pragma region "Template base"
template <typename T>
class ReduceTmpl {
public:
    using DataType = T;

public:
    __aicore__ inline ReduceTmpl() {}
};
#pragma endregion
#endif
} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops
#endif