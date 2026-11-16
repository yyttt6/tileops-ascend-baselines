/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT4_E2M1_H
#define OP_API_FLOAT4_E2M1_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP4 E2M1 floating-point format (OCP MX MXFP4)
 *
 * Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
 *         +---+----+-+
 *         | S | EE |M|
 *         +---+----+-+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 1
 * - Max normal value: 6.0 (0x7 = 0 11 1, exp=3, man=1 -> 2^2 * 1.5 = 6.0)
 * - Min normal value: -6.0 (0xF = 1 11 1)
 * - Min positive normal: 2^0 = 1.0 (0x2 = 0 01 0)
 * - Min positive denorm: 2^(1-bias) * (man/2) = 2^0 * 0.5 = 0.5 (0x1 = 0 00 1)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float4E2M1 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E2M1 format constants =============
    static constexpr uint8_t HIGHEST_VALUE = 0x07;        // 0 11 1 = 6.0 (max value)
    static constexpr uint8_t LOWEST_VALUE = 0x0F;         // 1 11 1 = -6.0 (min value)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x02; // 0 01 0 = 1.0 (min positive normal)
    static constexpr uint8_t BITS_MASK = 0x0F;            // 4-bit effective mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;        // 0 00 1 = 0.5 (epsilon)

    // Default constructor - initialize to zero
    constexpr Float4E2M1() : value(0) {}

    constexpr Float4E2M1(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK) {}

    Float4E2M1(float v);

    operator float() const;
    operator double() const;

    static constexpr Float4E2M1 Epsilon()
    {
        // 2^-1 = 0.5 (the difference between 1.0 and next representable value)
        return Float4E2M1(EPSILON_VALUE, FromBits()); // 0 00 1 = denorm 0.5
    }

    static constexpr Float4E2M1 Highest()
    {
        // exp=3, man=1 -> 2^2 * 1.5 = 6.0 (0 11 1 = 0x7)
        // OCP MX E2M1 has no NaN, all bit patterns are valid
        return Float4E2M1(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float4E2M1 Lowest()
    {
        // 1 11 1 = -6.0 (exp=3, man=1)
        return Float4E2M1(LOWEST_VALUE, FromBits());
    }

    static constexpr Float4E2M1 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 01 0)
        return Float4E2M1(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float4E2M1ToFloat(Float4E2M1 fp4);
    static Float4E2M1 FloatToFloat4E2M1(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float4E2M1& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float4E2M1& a [[maybe_unused]])
{
    return false; // E2M1 has no infinity
}

inline bool isnan(const op::Float4E2M1& a [[maybe_unused]])
{
    return false; // OCP MX E2M1 has no NaN
}

inline bool isfinite(const op::Float4E2M1& a [[maybe_unused]])
{
    return true; // All bit patterns are valid finite numbers
}

template <>
class numeric_limits<op::Float4E2M1> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false; // OCP MX E2M1 has no NaN
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 1;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = -2; // 2^(-2) for denorm
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 2; // 2^2 for normal
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        return 0.5f; // 2^0 * (1/2) = 0.5 (man=1, exp=0)
    }

    static constexpr op::Float4E2M1 min() { return op::Float4E2M1::MinPositiveNormal(); }
    static constexpr op::Float4E2M1 lowest() { return op::Float4E2M1::Lowest(); }
    static constexpr op::Float4E2M1 max() { return op::Float4E2M1::Highest(); }
    static constexpr op::Float4E2M1 epsilon() { return op::Float4E2M1::Epsilon(); }
    static constexpr op::Float4E2M1 infinity()
    {
        return op::Float4E2M1::Highest(); // No infinity, use max
    }
    static constexpr op::Float4E2M1 quiet_NaN()
    {
        // OCP MX E2M1 has no NaN, return zero as fallback
        return op::Float4E2M1(0x00, op::Float4E2M1::FromBits());
    }
    static constexpr op::Float4E2M1 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT4_E2M1_H
