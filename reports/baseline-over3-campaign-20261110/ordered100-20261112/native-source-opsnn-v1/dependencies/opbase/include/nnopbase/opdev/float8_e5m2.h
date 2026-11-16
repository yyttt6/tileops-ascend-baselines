/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E5M2_H
#define OP_API_FLOAT8_E5M2_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP8 E5M2 floating-point format (OCP standard)
 *
 * Format: 1 sign bit, 5 exponent bits, 2 mantissa bits
 *         +---+-----+--+
 *         | S | EEEEE|MM|
 *         +---+-----+--+
 *
 * - Bias: 15
 * - Max normal value: 57344 (0x7B)
 * - Min positive normal: 2^-14 = 0.00006103515625 (0x04)
 * - Supports Infinity (0x7C, 0xFC)
 * - Supports NaN (0x7D-0x7F, 0xFD-0xFF)
 */
struct Float8E5M2 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    static constexpr uint8_t INF_VALUE = 0x7C;            // 0 11111 00
    static constexpr uint8_t NAN_VALUE = 0x7F;            // 0 11111 11
    static constexpr uint8_t HIGHEST_VALUE = 0x7B;        // 0 11110 11 (max finite: 57344)
    static constexpr uint8_t LOWEST_VALUE = 0xFB;         // 1 11110 11 (min finite: -57344)
    static constexpr uint8_t EPSILON_VALUE = 0x34;        // 0 01101 00 (epsilon at 1.0)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04; // 0 00001 00 (min positive normal)

    // Default constructor - initialize to zero
    constexpr Float8E5M2() : value(0) {}

    constexpr Float8E5M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits) {}

    Float8E5M2(float v);

    operator float() const;
    operator double() const;

    static constexpr Float8E5M2 Epsilon()
    {
        // 2^-2 = 0.25 at exp = 15 (1.0)
        return Float8E5M2(EPSILON_VALUE, FromBits()); // 0 01101 00 = 1.0
    }

    static constexpr Float8E5M2 Highest()
    {
        // 57344.0 = 0x7B (0 11110 11)
        return Float8E5M2(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float8E5M2 Lowest()
    {
        // -57344.0 = 0xFB (1 11110 11)
        return Float8E5M2(LOWEST_VALUE, FromBits());
    }

    static constexpr Float8E5M2 MinPositiveNormal()
    {
        // 2^-14 = 0.00006103515625 = 0x04 (0 00001 00)
        return Float8E5M2(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float8E5M2ToFloat(Float8E5M2 fp8);
    static Float8E5M2 FloatToFloat8E5M2(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float8E5M2& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float8E5M2& a) { return a.IsInf(); }

inline bool isnan(const op::Float8E5M2& a) { return a.IsNaN(); }

inline bool isfinite(const op::Float8E5M2& a) { return !a.IsNaN() && !a.IsInf(); }

template <>
class numeric_limits<op::Float8E5M2> {
public:
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 2;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = -14;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        return 0.0000152587890625f; // 2^-16
    }

    static constexpr op::Float8E5M2 min() { return op::Float8E5M2::MinPositiveNormal(); }
    static constexpr op::Float8E5M2 lowest() { return op::Float8E5M2::Lowest(); }
    static constexpr op::Float8E5M2 max() { return op::Float8E5M2::Highest(); }
    static constexpr op::Float8E5M2 epsilon() { return op::Float8E5M2::Epsilon(); }
    static constexpr op::Float8E5M2 infinity()
    {
        return op::Float8E5M2(op::Float8E5M2::INF_VALUE, op::Float8E5M2::FromBits());
    }
    static constexpr op::Float8E5M2 quiet_NaN()
    {
        return op::Float8E5M2(op::Float8E5M2::NAN_VALUE, op::Float8E5M2::FromBits());
    }
    static constexpr op::Float8E5M2 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT8_E5M2_H
