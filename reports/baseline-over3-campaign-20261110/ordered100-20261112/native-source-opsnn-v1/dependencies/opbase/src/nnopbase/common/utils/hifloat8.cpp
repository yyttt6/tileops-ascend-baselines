/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/hifloat8.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "securec.h"

#include "opdev/op_log.h"

namespace {
// Exponent to mantissa width mapping
constexpr std::array<uint8_t, 24> HIF8_EXP_TO_MANTISSA_WIDTH = {{
    3,
    3,
    3,
    3, // [0, 3]
    2,
    2,
    2,
    2, // [4, 7]
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1, // [8, 15]
    0,
    0,
    0,
    0,
    0,
    0,
    0,                        // [16, 22]
    static_cast<uint8_t>(-1), // [23]
}};

// Exponent to Dot prefix mapping
constexpr std::array<uint8_t, 16> HIF8_EXP_TO_DOT = {{
    0b0001, // [0]
    0b0010, // [1]
    0b0100,
    0b0100, // [2, 3]
    0b1000,
    0b1000,
    0b1000,
    0b1000, // [4, 7]
    0b1100,
    0b1100,
    0b1100,
    0b1100,
    0b1100,
    0b1100,
    0b1100,
    0b1100, // [8, 15]
}};

// Bit width constants
constexpr uint8_t HIF8_BITS = 8;                                  // HiF8 total bits
constexpr uint8_t HIF8_MSB_INDEX = HIF8_BITS - 1;                 // HiF8 MSB index (7)
constexpr uint8_t UINT32_BITS = 32;                               // uint32_t bit width
constexpr uint8_t CLZ_OFFSET_FOR_UINT8 = UINT32_BITS - HIF8_BITS; // __builtin_clz offset for uint8_t (24)

constexpr uint8_t HIF8_DML_MASK = 0b01111000;
constexpr uint8_t HIF8_DML_FLAG = static_cast<uint8_t>(-1);
constexpr float EXP_BASE = 2.0F;

constexpr int8_t HIF8_DML_EXP_MAX = -16;
constexpr int8_t HIF8_DML_EXP_BIAS = 23;
constexpr uint8_t HIF8_DOT_OFFSET = 3;

constexpr int8_t HIF8_NML_EXP_MAX = 15;
constexpr int8_t HIF8_DML_EXP_MIN = -22;

constexpr uint8_t HIF8_SIGN_MASK = 0b10000000; // Sign mask for external operators

constexpr uint8_t IEEE_FP32_EXP_BITS = 8;
constexpr uint8_t IEEE_FP32_MAN_BITS = 32 - 1 - IEEE_FP32_EXP_BITS;

// Bit extraction utility
uint32_t Extract32(uint32_t u32, uint8_t width, uint8_t offset) { return (u32 >> offset) & ((1U << width) - 1U); }

// Bit deposition utility
uint32_t Deposit32(uint32_t u32, uint8_t width, uint8_t offset, uint32_t val)
{
    const uint32_t mask = (1U << width) - 1U;
    return (u32 & ~(mask << offset)) | ((val & mask) << offset);
}

// Encode exponent and mantissa to HiF8 format
uint8_t EncodeHiF8(int8_t exp, uint8_t mantissa)
{
    if (exp <= HIF8_DML_EXP_MAX) {
        return static_cast<uint8_t>(exp + HIF8_DML_EXP_BIAS);
    }
    const uint8_t mag = static_cast<uint8_t>(exp < 0 ? -exp : exp);
    auto encode_em = [](uint8_t m, uint8_t se_bit) -> uint8_t {
        if (m == 0) {
            return 0;
        }
        // Replace the most-significant bit in `m` to `se_bit`.
        return Deposit32(m, 1, HIF8_MSB_INDEX - (__builtin_clz(m) - CLZ_OFFSET_FOR_UINT8), se_bit);
    };
    return static_cast<uint8_t>(HIF8_EXP_TO_DOT[mag] << HIF8_DOT_OFFSET) |
           static_cast<uint8_t>(encode_em(mag, exp < 0) << HIF8_EXP_TO_MANTISSA_WIDTH[mag]) | mantissa;
}

// IEEE floating point type classification
enum class IeeeType : int8_t {
    ZERO,
    NOT_A_NUMBER,
    POSITIVE_INFINITY,
    ORDINARY,
};

// Unpack IEEE floating point value
template <uint8_t exp_bits, uint8_t mantissa_bits>
static IeeeType UnpackIeeeFp(uint32_t bits, uint8_t* sign_bit, int8_t* exp, uint32_t* mantissa)
{
    constexpr uint32_t ieee_sign_bit = 1U << (exp_bits + mantissa_bits);
    constexpr int8_t bias = static_cast<int8_t>((1U << (exp_bits - 1U)) - 1U);

    if ((bits & ~ieee_sign_bit) == 0) {
        return IeeeType::ZERO;
    }

    *sign_bit = (bits & ieee_sign_bit) != 0 ? HIF8_SIGN_MASK : 0U;
    *exp = static_cast<int8_t>(Extract32(bits, exp_bits, mantissa_bits)) - bias;
    *mantissa = Extract32(bits, mantissa_bits, 0);

    // Handle NaN or Inf.
    if (static_cast<uint8_t>(*exp) == (1U << (exp_bits - 1U))) {
        return *mantissa != 0 ? IeeeType::NOT_A_NUMBER : IeeeType::POSITIVE_INFINITY;
    }
    // Handle IEEE subnormal values.
    if (*exp == -bias && *mantissa != 0) {
        const uint8_t shift = mantissa_bits - (UINT32_BITS - __builtin_clz(*mantissa)) + 1;
        *exp = -bias + 1 - shift;
        *mantissa = Extract32(*mantissa << shift, mantissa_bits, 0);
    }
    return IeeeType::ORDINARY;
}

// Round mantissa to specified width
template <uint8_t mantissa_bits>
static bool RoundMantissa(uint32_t* mantissa, uint32_t probe_bit)
{
    const uint32_t tmp = *mantissa + probe_bit;
    *mantissa = Extract32(tmp, mantissa_bits, 0);
    return (tmp & (1U << mantissa_bits)) != 0;
}

// Convert IEEE floating point bits to HiF8
template <uint8_t exp_bits, uint8_t mantissa_bits>
static uint8_t HiF8FromIeeeBits(uint32_t bits)
{
    uint8_t sign_bit = 0;
    int8_t exp = 0;
    uint32_t mantissa = 0;
    const auto type = UnpackIeeeFp<exp_bits, mantissa_bits>(bits, &sign_bit, &exp, &mantissa);
    if (type == IeeeType::ZERO || (type == IeeeType::ORDINARY && exp < HIF8_DML_EXP_MIN - 1)) {
        return 0;
    } else if (type == IeeeType::NOT_A_NUMBER) {
        return op::HiFloat8::HIF8_NAN_VALUE;
    } else if (type == IeeeType::POSITIVE_INFINITY || (type == IeeeType::ORDINARY && exp > HIF8_NML_EXP_MAX)) {
        return sign_bit | op::HiFloat8::HIF8_INF_VALUE;
    }

    // TA rounding.
    const uint8_t bits_to_discard = mantissa_bits - HIF8_EXP_TO_MANTISSA_WIDTH[static_cast<ssize_t>(std::abs(exp))];
    if (RoundMantissa<mantissa_bits>(&mantissa, 1U << (bits_to_discard - 1U))) {
        exp += 1;
        // Handle overflow after TA rounding.
        if (exp > HIF8_NML_EXP_MAX) {
            return sign_bit | op::HiFloat8::HIF8_INF_VALUE;
        }
    }
    return sign_bit | EncodeHiF8(exp, mantissa >> bits_to_discard);
}

// Bit cast utility
template <typename To, typename From>
auto BitCast(const From& src) -> To
{
    static_assert(sizeof(To) == sizeof(From), "");
    To dst = 0;
    OP_CHECK(memcpy_s(&dst, sizeof(To), &src, sizeof(From)) == EOK, OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
             ;);
    return dst;
}

// Extract exponent from encoded value
int8_t ExtractExponent(uint8_t dot, uint8_t mantissa_width, uint8_t raw)
{
    if (dot == 0) {
        return 0;
    }
    const uint8_t em = Extract32(raw, dot, mantissa_width);
    const uint8_t se = em & static_cast<uint8_t>(1U << (dot - 1U));
    const uint8_t mag = Deposit32(em, 1, dot - 1U, 1);
    return (se != 0 ? -1 : 1) * static_cast<int8_t>(mag);
}

} // anonymous namespace

namespace op {
HiFloat8::HiFloat8(float f32) : value(BitsFromFp32(BitCast<uint32_t>(f32))) {}

HiFloat8::operator float() const { return Hifp8ToFloat(*this); }

bool HiFloat8::IsNaN() const { return value == HIF8_NAN_VALUE; }

bool HiFloat8::IsInf() const { return (value & static_cast<uint8_t>(~HIF8_SIGN_MASK)) == HIF8_INF_VALUE; }

bool HiFloat8::IsZero() const { return (value & static_cast<uint8_t>(~HIF8_SIGN_MASK)) == HIF8_ZERO_VALUE; }

uint8_t HiFloat8::BitsFromFp32(uint32_t f32) { return HiF8FromIeeeBits<IEEE_FP32_EXP_BITS, IEEE_FP32_MAN_BITS>(f32); }

float HiFloat8::Hifp8ToFloat(HiFloat8 hif8)
{
    if (hif8.IsNaN()) {
        return NAN;
    } else if (hif8.IsInf()) {
        return (hif8.value & HIF8_SIGN_MASK) != 0 ? -std::numeric_limits<float>::infinity() :
                                                    std::numeric_limits<float>::infinity();
    }
    static const std::array<std::pair<uint8_t, uint8_t>, 16> DOT_TABLE = {{
        {HIF8_DML_FLAG, 4}, // 0b0000
        {0, 4},             // 0b0001
        {1, 3},
        {1, 3}, // 0b001.
        {2, 2},
        {2, 2},
        {2, 2},
        {2, 2}, // 0b01..
        {3, 2},
        {3, 2},
        {3, 2},
        {3, 2}, // 0b10..
        {4, 2},
        {4, 2},
        {4, 2},
        {4, 2}, // 0b11..
    }};
    const float sign = (hif8.value & HIF8_SIGN_MASK) != 0 ? -1.0F : 1.0F;
    const auto pair = DOT_TABLE[(hif8.value & HIF8_DML_MASK) >> HIF8_DOT_OFFSET];
    const uint8_t dot = pair.first;
    const uint8_t width = pair.second;

    if (dot == HIF8_DML_FLAG) {
        const int8_t dml_exp = static_cast<int8_t>(hif8.value &
                                                   static_cast<uint8_t>(~(HIF8_SIGN_MASK | HIF8_DML_MASK)));
        if (dml_exp == 0) {
            return 0.0F;
        }
        return sign * powf(EXP_BASE, static_cast<float>(dml_exp - HIF8_DML_EXP_BIAS));
    }

    const uint8_t mantissa_width = HIF8_MSB_INDEX - (dot + width);
    float mantissa_value = 0.0F;
    for (uint8_t n = 0; n < mantissa_width; n++) {
        if ((hif8.value & static_cast<uint8_t>(1U << n)) == 0) {
            continue;
        }
        mantissa_value += powf(EXP_BASE, -static_cast<float>(mantissa_width - n));
    }

    const int8_t exp = ExtractExponent(dot, mantissa_width, hif8.value);
    return sign * powf(EXP_BASE, static_cast<float>(exp)) * (1.0F + mantissa_value);
}

} // namespace op
