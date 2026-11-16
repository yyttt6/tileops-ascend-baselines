/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT6_E3M2_H
#define OP_API_FLOAT6_E3M2_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP6 E3M2 floating-point format (OCP MX MXFP6)
 *
 * Format: 1 sign bit, 3 exponent bits, 2 mantissa bits
 *         +---+-----+--+
 *         | S | EEE |MM|
 *         +---+-----+--+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 3
 * - Max normal value: 28.0 (0x1F = 0 111 11, exp=7, man=3 -> 2^4 * 1.75 = 28.0)
 * - Min normal value: -28.0 (0x3F = 1 111 11)
 * - Min positive normal: 2^-2 = 0.25 (0x04 = 0 001 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 0.0625 (0x01 = 0 000 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float6E3M2 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E3M2 format constants (OCP MX MXFP6) =============
    static constexpr uint8_t BITS_MASK = 0x3F;            // 6-bit mask
    static constexpr uint8_t HIGHEST_VALUE = 0x1F;        // 0 111 11 (max: 28.0)
    static constexpr uint8_t LOWEST_VALUE = 0x3F;         // 1 111 11 (min: -28.0)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04; // 0 001 00 (min positive normal: 0.25)
    static constexpr uint8_t EPSILON_VALUE = 0x04;        // 0 001 00 (epsilon at 1.0: 0.25)

    // Default constructor - initialize to zero
    constexpr Float6E3M2() : value(0) {}

    constexpr Float6E3M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK) {}

    Float6E3M2(float v);

    operator float() const;

    operator double() const;

    static constexpr Float6E3M2 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return Float6E3M2(EPSILON_VALUE, FromBits()); // 0 001 00 = 2^(1-3) * 1.0 = 0.25
    }

    static constexpr Float6E3M2 Highest()
    {
        // 28.0 = 0x1F (0 111 11) -> exp=7, man=3 -> 2^4 * 1.75 = 28.0
        // OCP MX E3M2 has no NaN, all bit patterns are valid
        return Float6E3M2(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float6E3M2 Lowest()
    {
        return Float6E3M2(LOWEST_VALUE, FromBits()); // 1 111 11 = -28.0
    }

    static constexpr Float6E3M2 MinPositiveNormal()
    {
        // 2^-2 = 0.25 (0 001 00)
        return Float6E3M2(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float6E3M2ToFloat(Float6E3M2 fp6);
    static Float6E3M2 FloatToFloat6E3M2(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float6E3M2& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float6E3M2& a [[maybe_unused]])
{
    return false; // E3M2 has no infinity
}

inline bool isnan(const op::Float6E3M2& a [[maybe_unused]])
{
    return false; // OCP MX E3M2 has no NaN
}

inline bool isfinite(const op::Float6E3M2& a [[maybe_unused]])
{
    return true; // All bit patterns are valid finite numbers
}

template <>
class numeric_limits<op::Float6E3M2> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false; // OCP MX E3M2 has no NaN
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
    static constexpr int min_exponent = -2;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 4;
    static constexpr int max_exponent10 = 1;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        // OCP MX E3M2: min subnorm = S 000 01 = 2^(1-bias) * (0 + M/4) = 2^(-2) * 0.25 = 0.0625
        // Formula: (-1)^S * 2^(1-3) * (M/4) = M/16 where bias=3
        // When M=1: 1/16 = 0.0625
        return 0.0625f;
    }

    static constexpr op::Float6E3M2 min() { return op::Float6E3M2::MinPositiveNormal(); }
    static constexpr op::Float6E3M2 lowest() { return op::Float6E3M2::Lowest(); }
    static constexpr op::Float6E3M2 max() { return op::Float6E3M2::Highest(); }
    static constexpr op::Float6E3M2 epsilon() { return op::Float6E3M2::Epsilon(); }
    static constexpr op::Float6E3M2 infinity()
    {
        return op::Float6E3M2::Highest(); // No infinity, use max
    }
    static constexpr op::Float6E3M2 quiet_NaN()
    {
        // OCP MX E3M2 has no NaN, return zero as fallback
        return op::Float6E3M2(0x00, op::Float6E3M2::FromBits());
    }
    static constexpr op::Float6E3M2 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT6_E3M2_H
