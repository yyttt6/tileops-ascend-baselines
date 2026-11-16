/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float8_e8m0.h"

#include <cmath>
#include <cstdint>

namespace {
// ============= FP32 format constants =============
constexpr int FP32_MAN_LEN_VAL = 23;
constexpr int FP32_SIGN_SHIFT_VAL = 31;
constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
constexpr uint32_t FP32_ROUND_THRESHOLD = 0x400000; // 0.5 in mantissa representation

// ============= E8M0 format constants (unsigned, 8 exponent bits) =============
constexpr int MAN_BITS = 0;
constexpr int EXP_BIAS = 127;
constexpr uint8_t EXP_MASK = 0xFF;              // 11111111 (all 8 bits for exponent)
constexpr uint8_t MAX_EXP_VAL = 0xFE;           // 254 (max normal exponent)
constexpr uint8_t NAN_EXP_VAL = 0xFF;           // 255 (NaN)
constexpr uint8_t ZERO_EXP_VAL = 0x00;          // 0 (zero)
constexpr uint8_t ZERO_VALUE = 0x00;            // 0 (zero)
constexpr uint8_t EXP_OVERFLOW_THRESHOLD = 254; // max allowed exponent before overflow
} // anonymous namespace

namespace op {

Float8E8M0::Float8E8M0(float v) : value(FloatToFloat8E8M0(v).value) {}

Float8E8M0::operator float() const { return Float8E8M0ToFloat(*this); }

Float8E8M0::operator double() const { return static_cast<double>(Float8E8M0ToFloat(*this)); }

bool Float8E8M0::IsZero() const { return value == ZERO_VALUE; }

bool Float8E8M0::IsNaN() const
{
    // NaN: exponent = 255
    return value == NAN_VALUE;
}

bool Float8E8M0::IsInf() const
{
    // E8M0 has no infinity encoding, only NaN at 0xFF
    return false;
}

// Get the exponent value (unbiased)
int32_t Float8E8M0::GetExponent() const
{
    if (IsZero() || IsNaN() || IsInf()) {
        return 0;
    }
    return static_cast<int32_t>(value) - EXP_BIAS;
}

float Float8E8M0::Float8E8M0ToFloat(Float8E8M0 fp8)
{
    if (fp8.IsZero()) {
        return 0.0f;
    }

    if (fp8.IsNaN()) {
        return std::nanf("");
    }

    // E8M0 has no infinity encoding, only NaN at 0xFF

    uint32_t exp = fp8.value;

    // E8M0: value = 2^(exp - 127)
    // FP32: value = 2^(exp - 127) with mantissa = 0
    // For E8M0, the FP32 exponent equals the E8M0 exponent (same bias)

    union {
        uint32_t u;
        float f;
    } result;
    result.u = (exp << FP32_MAN_LEN_VAL); // sign=0, exponent=exp, mantissa=0
    return result.f;
}

Float8E8M0 Float8E8M0::FloatToFloat8E8M0(float f)
{
    if (std::isnan(f)) {
        return Float8E8M0(NAN_VALUE, FromBits());
    }

    if (std::fpclassify(f) == FP_ZERO) {
        return Float8E8M0(ZERO_VALUE, FromBits());
    }

    // E8M0 is unsigned, negative values are not representable
    if (f < 0.0f) {
        return Float8E8M0(NAN_VALUE, FromBits());
    }

    // E8M0 has no infinity encoding, clamp to max value (0xFE)
    if (std::isinf(f)) {
        return Float8E8M0(HIGHEST_VALUE, FromBits());
    }

    // E8M0 can only represent powers of 2
    // Find the exponent such that 2^exp is closest to f
    union {
        uint32_t u;
        float f;
    } fp32;
    fp32.f = f;

    uint32_t fp32Exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
    uint32_t fp32Man = fp32.u & FP32_MAN_MASK_23BIT;

    // Round to nearest power of 2
    // If mantissa >= 0x400000 (0.5), round up the exponent
    if (fp32Man >= FP32_ROUND_THRESHOLD) {
        fp32Exp++;
    }

    // Handle overflow
    if (fp32Exp > EXP_OVERFLOW_THRESHOLD) {
        // E8M0 has no infinity encoding, clamp to max value (0xFE)
        return Float8E8M0(HIGHEST_VALUE, FromBits());
    }

    // Handle underflow (fp32Exp == 0 means denormal or zero)
    if (fp32Exp == 0) {
        return Float8E8M0(ZERO_VALUE, FromBits());
    }

    // E8M0 stores only the exponent (same bias as FP32)
    return Float8E8M0(static_cast<uint8_t>(fp32Exp), FromBits());
}

} // namespace op
