/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float8_e5m2.h"

#include <cmath>
#include <cstdint>

namespace {
// FP32 format constants
constexpr int FP32_EXP_BIAS_VAL = 127;
constexpr int FP32_MAN_LEN_VAL = 23;
constexpr int FP32_SIGN_SHIFT_VAL = 31;
constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;

// E5M2 format constants
constexpr int EXP_BITS = 5;
constexpr int MAN_BITS = 2;
constexpr int EXP_BIAS = 15;
constexpr int SIGN_SHIFT = 7;
constexpr uint8_t SIGN_MASK = 0x80;      // 1 00000 00
constexpr uint8_t EXP_MASK = 0x7C;       // 0 11111 00
constexpr uint8_t MAN_MASK = 0x03;       // 0 00000 11
constexpr uint8_t ABS_VALUE_MASK = 0x7F; // 0 11111 11 (exclude sign bit)
constexpr uint8_t MAX_EXP = 0x1F;        // 11111
constexpr uint8_t NEG_INF_VALUE = 0xFC;  // 1 11111 00
} // anonymous namespace

namespace op {

Float8E5M2::Float8E5M2(float v) : value(FloatToFloat8E5M2(v).value) {}

Float8E5M2::operator float() const { return Float8E5M2ToFloat(*this); }

Float8E5M2::operator double() const { return static_cast<double>(Float8E5M2ToFloat(*this)); }

bool Float8E5M2::IsZero() const { return (value & ABS_VALUE_MASK) == 0; }

bool Float8E5M2::IsNaN() const { return (value & EXP_MASK) == EXP_MASK && (value & MAN_MASK) != 0; }

bool Float8E5M2::IsInf() const { return value == INF_VALUE || value == NEG_INF_VALUE; }

float Float8E5M2::Float8E5M2ToFloat(Float8E5M2 fp8)
{
    if (fp8.IsZero()) {
        return (fp8.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    if (fp8.IsNaN()) {
        return std::nanf("");
    }

    if (fp8.IsInf()) {
        return (fp8.value & SIGN_MASK) != 0 ? -std::numeric_limits<float>::infinity() :
                                              std::numeric_limits<float>::infinity();
    }

    uint32_t sign = (fp8.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp8.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp8.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // FP8 E5M2: 1 sign, 5 exp (bias 15), 2 man

    int32_t fp32Exp = static_cast<int32_t>(exp) - EXP_BIAS + FP32_EXP_BIAS_VAL;
    uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

    if (exp == 0) {
        // Denormalized number
        if (man != 0) {
            // Normalize: shift mantissa left until leading 1
            int shift = 0;
            while ((man & (1 << MAN_BITS)) == 0) {
                man <<= 1;
                shift++;
            }
            man &= MAN_MASK; // Remove implicit leading 1
            fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
            fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);
        }
    } else {
        // Normal number - add implicit leading 1
        fp32Man |= FP32_IMPLICIT_1_VAL;
    }

    FP32 result;
    result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) |
               (fp32Man & FP32_MAN_MASK_23BIT);
    return result.f;
}

Float8E5M2 Float8E5M2::FloatToFloat8E5M2(float f)
{
    if (std::isnan(f)) {
        return Float8E5M2(NAN_VALUE, FromBits());
    }

    FP32 fp32;
    fp32.f = f;
    uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
    uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
    uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

    if (exp == 0 && man == 0) {
        // Zero
        return Float8E5M2(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
    }

    if (std::isinf(f)) {
        // E5M2 supports infinity
        return Float8E5M2(static_cast<uint8_t>((sign << SIGN_SHIFT) | INF_VALUE), FromBits());
    }

    // Calculate E5M2 exponent
    int32_t fp8Exp;
    uint32_t fp8Man;

    if (exp == 0) {
        // Denormalized FP32 input
        fp8Exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
        // Normalize the FP32 denormal
        while ((man & FP32_IMPLICIT_1_VAL) == 0) {
            man <<= 1;
            fp8Exp--;
        }
        man &= FP32_MAN_MASK_23BIT;
    } else {
        fp8Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
    }

    // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
    int mantissaShift = FP32_MAN_LEN_VAL - MAN_BITS;
    uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
    uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) != 0 ? 1 : 0;
    uint32_t manLsb = (man >> mantissaShift) & 1;

    fp8Man = man >> mantissaShift;
    if ((manRoundBit != 0) && ((manStickyBits != 0) || (manLsb != 0))) {
        fp8Man++;
        if (fp8Man > MAN_MASK) {
            fp8Man = 0;
            fp8Exp++;
        }
    }

    // Handle overflow/underflow
    if (fp8Exp <= 0) {
        // Underflow to zero or denormal
        if (fp8Exp < -MAN_BITS) {
            return Float8E5M2(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }
        // Create denormal
        fp8Man = (fp8Man | (1 << MAN_BITS)) >> (1 - fp8Exp);
        fp8Exp = 0;
    } else if (fp8Exp >= static_cast<int32_t>(MAX_EXP)) {
        // Overflow to infinity
        return Float8E5M2(static_cast<uint8_t>((sign << SIGN_SHIFT) | INF_VALUE), FromBits());
    }

    uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp8Exp << MAN_BITS) | (fp8Man & MAN_MASK));
    return Float8E5M2(result, FromBits());
}

} // namespace op
