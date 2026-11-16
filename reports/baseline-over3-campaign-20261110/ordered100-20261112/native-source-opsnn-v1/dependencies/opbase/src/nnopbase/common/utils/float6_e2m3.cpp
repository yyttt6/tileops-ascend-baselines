/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float6_e2m3.h"

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

// E2M3 format constants
// Bit layout: bit5=sign, bits4-3=exp, bits2-0=man
constexpr int EXP_BITS = 2;
constexpr int MAN_BITS = 3;
constexpr int EXP_BIAS = 1;
constexpr int SIGN_SHIFT = 5;
constexpr uint8_t SIGN_MASK = 0x20;      // 1 00 000 (bit 5)
constexpr uint8_t EXP_MASK = 0x18;       // 0 11 000 (bits 4-3)
constexpr uint8_t MAN_MASK = 0x07;       // 0 00 111 (bits 2-0)
constexpr uint8_t MAX_EXP = 0x03;        // 11
constexpr uint8_t ABS_VALUE_MASK = 0x1F; // 0 11111 (exclude sign bit)
} // anonymous namespace

namespace op {

Float6E2M3::Float6E2M3(float v) : value(FloatToFloat6E2M3(v).value) {}

Float6E2M3::operator float() const { return Float6E2M3ToFloat(*this); }

Float6E2M3::operator double() const { return static_cast<double>(Float6E2M3ToFloat(*this)); }

bool Float6E2M3::IsZero() const
{
    return (value & ABS_VALUE_MASK) == 0; // 排除符号位
}

bool Float6E2M3::IsNaN() const
{
    // OCP MX E2M3 has no NaN representation
    // All bit patterns represent valid numbers
    return false;
}

bool Float6E2M3::IsInf() const
{
    // E2M3 has no infinity representation
    return false;
}

float Float6E2M3::Float6E2M3ToFloat(Float6E2M3 fp6)
{
    if (fp6.IsZero()) {
        return (fp6.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    // OCP MX E2M3 has no NaN, all bit patterns are valid numbers

    uint32_t sign = (fp6.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp6.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp6.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // FP6 E2M3: 1 sign, 2 exp (bias 1), 3 man

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

Float6E2M3 Float6E2M3::FloatToFloat6E2M3(float f)
{
    // OCP MX E2M3 has no NaN, clamp NaN to positive max value (7.5)
    // NaN has no meaningful sign, so always clamp to +7.5
    if (std::isnan(f)) {
        // 0x1F = 0 11 111 = 2^2 * 1.875 = 7.5
        return Float6E2M3(HIGHEST_VALUE, FromBits());
    }

    FP32 fp32;
    fp32.f = f;
    uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
    uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
    uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

    if (exp == 0 && man == 0) {
        // Zero
        return Float6E2M3(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
    }

    if (std::isinf(f)) {
        // E2M3 has no infinity, clamp to max value
        // 0x1F = +7.5, 0x3F = -7.5
        return Float6E2M3(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    // Calculate E2M3 exponent
    int32_t fp6Exp;
    uint32_t fp6Man;

    if (exp == 0) {
        // Denormalized FP32 input
        fp6Exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
        while ((man & FP32_IMPLICIT_1_VAL) == 0) {
            man <<= 1;
            fp6Exp--;
        }
        man &= FP32_MAN_MASK_23BIT;
    } else {
        fp6Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
    }

    // Round mantissa from 23 bits to 3 bits with round-to-nearest-even
    int mantissaShift = FP32_MAN_LEN_VAL - MAN_BITS;
    uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
    uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) != 0 ? 1 : 0;
    uint32_t manLsb = (man >> mantissaShift) & 1;

    fp6Man = man >> mantissaShift;
    if ((manRoundBit != 0) && ((manStickyBits != 0) || (manLsb != 0))) {
        fp6Man++;
        if (fp6Man > MAN_MASK) {
            fp6Man = 0;
            fp6Exp++;
        }
    }

    // Handle overflow/underflow
    if (fp6Exp <= 0) {
        if (fp6Exp < -MAN_BITS) {
            return Float6E2M3(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }
        fp6Man = (fp6Man | (1 << MAN_BITS)) >> (1 - fp6Exp);
        fp6Exp = 0;
    } else if (fp6Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
        // Overflow - for E2M3, max exp is 3
        // OCP MX E2M3 has no NaN, clamp to max value (preserving sign)
        // 0x1F = 0 11 111 = +7.5, 0x3F = 1 11 111 = -7.5
        return Float6E2M3(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp6Exp << MAN_BITS) | (fp6Man & MAN_MASK));
    return Float6E2M3(result, FromBits());
}

} // namespace op
