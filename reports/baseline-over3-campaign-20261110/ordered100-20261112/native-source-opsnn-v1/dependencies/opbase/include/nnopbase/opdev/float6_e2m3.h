/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT6_E2M3_H
#define OP_API_FLOAT6_E2M3_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP6 E2M3 floating-point format (OCP MX MXFP6)
 *
 * Format: 1 sign bit, 2 exponent bits, 3 mantissa bits
 *         +---+----+---+
 *         | S | EE |MMM|
 *         +---+----+---+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 1
 * - Max normal value: 7.5 (0x1F = 0 11 111, exp=3, man=7 -> 2^2 * 1.875 = 7.5)
 * - Min normal value: -7.5 (0x3F = 1 11 111)
 * - Min positive normal: 2^-1 = 0.5 (0x08 = 0 01 000)
 * - Min positive denorm: 2^(1-bias) * (man/8) = 0.125 (0x01 = 0 00 001)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float6E2M3 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E2M3 format constants =============
    static constexpr uint8_t HIGHEST_VALUE = 0x1F;        // 0 11 111 = 7.5
    static constexpr uint8_t LOWEST_VALUE = 0x3F;         // 1 11 111 = -7.5
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x08; // 0 01 000 (min positive normal)
    static constexpr uint8_t BITS_MASK = 0x3F;            // 6-bit valid bits mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;        // 0 00 001 = denorm 0.125

    // Default constructor - initialize to zero
    constexpr Float6E2M3() : value(0) {}

    constexpr Float6E2M3(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK) {}

    Float6E2M3(float v);

    operator float() const;
    operator double() const;

    static constexpr Float6E2M3 Epsilon()
    {
        // 2^-3 = 0.125 (the difference between 1.0 and next representable value)
        return Float6E2M3(EPSILON_VALUE, FromBits()); // 0 00 001 = denorm 0.125
    }

    static constexpr Float6E2M3 Highest()
    {
        // exp=3, man=7 -> 2^2 * (1 + 7/8) = 4 * 1.875 = 7.5
        // OCP MX E2M3 has no NaN, all bit patterns are valid
        return Float6E2M3(HIGHEST_VALUE, FromBits()); // 0 11 111 = 7.5
    }

    static constexpr Float6E2M3 Lowest()
    {
        return Float6E2M3(LOWEST_VALUE, FromBits()); // 1 11 111 = -7.5
    }

    static constexpr Float6E2M3 MinPositiveNormal()
    {
        // 2^-1 = 0.5 (0 01 000)
        return Float6E2M3(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float6E2M3ToFloat(Float6E2M3 fp6);
    static Float6E2M3 FloatToFloat6E2M3(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float6E2M3& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float6E2M3& a [[maybe_unused]])
{
    return false; // E2M3 has no infinity
}

inline bool isnan(const op::Float6E2M3& a [[maybe_unused]])
{
    return false; // OCP MX E2M3 has no NaN
}

inline bool isfinite(const op::Float6E2M3& a [[maybe_unused]])
{
    return true; // All bit patterns are valid finite numbers
}

template <>
class numeric_limits<op::Float6E2M3> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false; // OCP MX E2M3 has no NaN
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 3;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = -1;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 2;
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        // OCP MX E2M3: min subnorm = S 00 001 = 2^0 * 0.125 = 0.125
        // Formula: (-1)^S * 2^(1-bias) * (M/8) = M/8 where bias=1
        // When M=1: 1/8 = 0.125
        return 0.125f;
    }

    static constexpr op::Float6E2M3 min() { return op::Float6E2M3::MinPositiveNormal(); }
    static constexpr op::Float6E2M3 lowest() { return op::Float6E2M3::Lowest(); }
    static constexpr op::Float6E2M3 max() { return op::Float6E2M3::Highest(); }
    static constexpr op::Float6E2M3 epsilon() { return op::Float6E2M3::Epsilon(); }
    static constexpr op::Float6E2M3 infinity()
    {
        return op::Float6E2M3::Highest(); // No infinity, use max
    }
    static constexpr op::Float6E2M3 quiet_NaN()
    {
        // OCP MX E2M3 has no NaN, return zero as fallback
        return op::Float6E2M3(0x00, op::Float6E2M3::FromBits());
    }
    static constexpr op::Float6E2M3 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT6_E2M3_H
