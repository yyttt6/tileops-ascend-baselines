/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float4_e2m1.h"

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

// E2M1 format constants
// Bit layout: bit3=sign, bits2-1=exp, bit0=man
constexpr int MAN_BITS = 1;
constexpr int EXP_BIAS = 1;
constexpr int SIGN_SHIFT = 3;
constexpr uint8_t SIGN_MASK = 0x08;      // 1 00 0 (bit 3)
constexpr uint8_t EXP_MASK = 0x06;       // 0 11 0 (bits 2-1)
constexpr uint8_t MAN_MASK = 0x01;       // 0 00 1 (bit 0)
constexpr uint8_t MAX_EXP = 0x03;        // 11
constexpr uint8_t ABS_VALUE_MASK = 0x07; // 0 111 (exclude sign bit)
} // anonymous namespace

namespace op {

Float4E2M1::Float4E2M1(float v) : value(FloatToFloat4E2M1(v).value) {}

Float4E2M1::operator float() const { return Float4E2M1ToFloat(*this); }

Float4E2M1::operator double() const { return static_cast<double>(Float4E2M1ToFloat(*this)); }

bool Float4E2M1::IsZero() const { return (value & ABS_VALUE_MASK) == 0; }

bool Float4E2M1::IsNaN() const
{
    // OCP MX E2M1 has no NaN representation
    // All bit patterns represent valid numbers
    return false;
}

bool Float4E2M1::IsInf() const
{
    // E2M1 has no infinity representation
    return false;
}

float Float4E2M1::Float4E2M1ToFloat(Float4E2M1 fp4)
{
    if (fp4.IsZero()) {
        return (fp4.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    // OCP MX E2M1 has no NaN, all bit patterns are valid numbers

    uint32_t sign = (fp4.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp4.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp4.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // FP4 E2M1: 1 sign, 2 exp (bias 1), 1 man

    int32_t fp32Exp = static_cast<int32_t>(exp) - EXP_BIAS + FP32_EXP_BIAS_VAL;
    uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

    if (exp == 0) {
        // Denormalized number
        if (man != 0) {
            // OCP MX E2M1 denorm: value = 2^(1-bias) × (man/2) = 2^0 × 0.5 = 0.5
            // FP32 表示: exp=126 对应 2^(-1), 配合隐含的 1.0 得到 0.5
            // man=1 -> 0.5
            fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - 1; // = 126, 对应 2^(-1)
            fp32Man = 0;                                    // FP32 隐含的 1.0 正好表示 E2M1 的 man/2 = 0.5
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

Float4E2M1 Float4E2M1::FloatToFloat4E2M1(float f)
{
    // OCP MX E2M1 has no NaN, clamp NaN to positive max value (6.0)
    // NaN has no meaningful sign, so always clamp to +6.0
    if (std::isnan(f)) {
        // 0x07 = 0 11 1 = 2^2 * 1.5 = 6.0
        return Float4E2M1(HIGHEST_VALUE, FromBits());
    }

    FP32 fp32;
    fp32.f = f;
    uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
    uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
    uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

    if (exp == 0 && man == 0) {
        // Zero
        return Float4E2M1(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
    }

    if (std::isinf(f)) {
        // E2M1 has no infinity, clamp to max value
        // 0x07 = 0 11 1 = 2^2 * 1.5 = 6.0, 0x0F = -6.0
        return Float4E2M1(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    // Calculate E2M1 exponent
    int32_t fp4Exp;
    uint32_t fp4Man;

    if (exp == 0) {
        // Denormalized FP32 input
        fp4Exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
        while ((man & FP32_IMPLICIT_1_VAL) == 0) {
            man <<= 1;
            fp4Exp--;
        }
        man &= FP32_MAN_MASK_23BIT;
    } else {
        fp4Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
    }

    // Round mantissa from 23 bits to 1 bit with round-to-nearest-even
    int mantissaShift = FP32_MAN_LEN_VAL - MAN_BITS;
    uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
    uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) != 0 ? 1 : 0;
    uint32_t manLsb = (man >> mantissaShift) & 1;

    fp4Man = man >> mantissaShift;
    if ((manRoundBit != 0) && ((manStickyBits != 0) || (manLsb != 0))) {
        fp4Man++;
        if (fp4Man > MAN_MASK) {
            fp4Man = 0;
            fp4Exp++;
        }
    }

    // Handle overflow/underflow
    if (fp4Exp <= 0) {
        if (fp4Exp < -MAN_BITS) {
            return Float4E2M1(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }
        fp4Man = (fp4Man | (1 << MAN_BITS)) >> (1 - fp4Exp);
        fp4Exp = 0;
    } else if (fp4Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
        // Overflow - for E2M1, max exp is 3
        // OCP MX E2M1 has no NaN, clamp to max value (preserving sign)
        // 0x07 = 0 11 1 = +6.0, 0x0F = 1 11 1 = -6.0
        return Float4E2M1(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp4Exp << MAN_BITS) | (fp4Man & MAN_MASK));
    return Float4E2M1(result, FromBits());
}

} // namespace op
