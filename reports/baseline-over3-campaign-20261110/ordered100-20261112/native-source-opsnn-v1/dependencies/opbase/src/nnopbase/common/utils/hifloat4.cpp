/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/hifloat4.h"

#include <cmath>
#include <cstdint>

namespace {
// ============= FP32 format constants =============
constexpr int FP32_EXP_BIAS_VAL = 127;
constexpr int FP32_MAN_LEN_VAL = 23;
constexpr int FP32_SIGN_SHIFT_VAL = 31;
constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;

// Constants for HiFloat4 format
// Bit layout: bit3=sign, bit2=exp, bits1-0=man
constexpr int MAN_BITS = 2;
constexpr int EXP_BIAS = 1;
constexpr uint8_t SIGN_MASK = 0x08;      // 1 0 00 (bit 3)
constexpr uint8_t EXP_MASK = 0x04;       // 0 1 00 (bit 2)
constexpr uint8_t MAN_MASK = 0x03;       // 0 0 11 (bits 1-0)
constexpr uint8_t MAX_EXP = 0x01;        // 1
constexpr int SIGN_SHIFT = 3;            // Sign bit position
constexpr uint8_t ABS_VALUE_MASK = 0x07; // Mask excluding sign bit
} // anonymous namespace

namespace op {

HiFloat4::HiFloat4(float v) : value(FloatToHiFloat4(v).value) {}

HiFloat4::operator float() const { return HiFloat4ToFloat(*this); }

HiFloat4::operator double() const { return static_cast<double>(HiFloat4ToFloat(*this)); }

bool HiFloat4::IsZero() const
{
    return (value & ABS_VALUE_MASK) == 0; // 排除符号位
}

bool HiFloat4::IsNaN() const
{
    // HiFloat4 has no NaN representation
    // All bit patterns represent valid numbers
    return false;
}

bool HiFloat4::IsInf() const
{
    // HiFloat4 has no infinity representation
    return false;
}

float HiFloat4::HiFloat4ToFloat(HiFloat4 fp4)
{
    if (fp4.IsZero()) {
        return (fp4.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    uint32_t sign = (fp4.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp4.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp4.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // HiFloat4: 1 sign, 1 exp (bias 1), 2 man

    int32_t fp32Exp;
    uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

    if (exp == 0) {
        // Denormalized number
        // HiFloat4: value = 2^(1-bias) * (man/4) = 2^0 * (man/4) = man/4
        if (man != 0) {
            // Denorm values: 0.25, 0.5, 0.75 for man=1,2,3
            // Find leading 1 position for normalization
            int shift = 0;
            uint32_t tempMan = man;
            while ((tempMan & (1 << MAN_BITS)) == 0 && shift < MAN_BITS) {
                tempMan <<= 1;
                shift++;
            }
            // Effective exponent for denorm is 1-bias=0
            // fp32Exp = FP32_BIAS + (1-bias) - shift = 127 - shift
            fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
            fp32Man = (tempMan & MAN_MASK) << (FP32_MAN_LEN_VAL - MAN_BITS);
        } else {
            fp32Exp = 0;
        }
    } else {
        // Normal number: value = 2^(1-1) * (1 + man/4) = (1 + man/4)
        fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL; // 127
        fp32Man |= FP32_IMPLICIT_1_VAL;             // Add implicit 1
    }

    FP32 result;
    result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) |
               (fp32Man & FP32_MAN_MASK_23BIT);
    return result.f;
}

HiFloat4 HiFloat4::FloatToHiFloat4(float f)
{
    // HiFloat4 has no NaN, clamp NaN to positive max value (1.75)
    // NaN has no meaningful sign, so always clamp to +1.75
    if (std::isnan(f)) {
        // 0x07 = 0 11 1 = 2^1 * 1.75 = 1.75
        return HiFloat4(HIGHEST_VALUE, FromBits());
    }

    FP32 fp32;
    fp32.f = f;
    uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
    uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
    uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

    if (exp == 0 && man == 0) {
        // Zero
        return HiFloat4(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
    }

    if (std::isinf(f)) {
        // HiFloat4 has no infinity, clamp to max value
        // 0x07 = +1.75, 0x0F = -1.75
        return HiFloat4(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    // Calculate HiFloat4 exponent
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

    // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
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
            return HiFloat4(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }
        fp4Man = (fp4Man | (1 << MAN_BITS)) >> (1 - fp4Exp);
        fp4Exp = 0;
    } else if (fp4Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
        // Overflow - for HiFloat4, max exp is 1
        // HiFloat4 has no NaN, clamp to max value (preserving sign)
        // 0x07 = +1.75, 0x0F = -1.75
        return HiFloat4(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp4Exp << MAN_BITS) | (fp4Man & MAN_MASK));
    return HiFloat4(result, FromBits());
}

} // namespace op
