/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT4_E1M2_H
#define OP_API_FLOAT4_E1M2_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP4 E1M2 floating-point format
 *
 * Format: 1 sign bit, 1 exponent bit, 2 mantissa bits
 *         +---+--+---+
 *         | S |E |MM |
 *         +---+--+---+
 *
 * Properties:
 * - Bias: 1
 * - Max normal value: 1.75 (0x7 = 0 1 11, exp=1, man=3 -> 2^0 * 1.75)
 * - Min normal value: -1.75 (0xF = 1 1 11, exp=1, man=3 -> -2^0 * 1.75)
 * - Min positive normal: 2^0 = 1.0 (0x4 = 0 1 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 2^0 * 0.25 = 0.25 (0x1 = 0 0 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float4E1M2 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E1M2 format constants =============
    static constexpr uint8_t BITS_MASK = 0x0F;            // 4-bit mask
    static constexpr uint8_t HIGHEST_VALUE = 0x07;        // 0 1 11 (max value: 1.75)
    static constexpr uint8_t LOWEST_VALUE = 0x0F;         // 1 1 11 (min value: -1.75)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04; // 0 1 00 (min positive normal)
    static constexpr uint8_t EPSILON_VALUE = 0x01;        // epsilon value

    // Default constructor - initialize to zero
    constexpr Float4E1M2() : value(0) {}

    constexpr Float4E1M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK) {}

    Float4E1M2(float v);

    operator float() const;
    operator double() const;

    static constexpr Float4E1M2 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return Float4E1M2(EPSILON_VALUE, FromBits()); // 0 0 01 = denorm 0.25
    }

    static constexpr Float4E1M2 Highest()
    {
        // exp=1, man=3 -> 2^0 * 1.75 = 1.75 (0 1 11 = 0x7)
        return Float4E1M2(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float4E1M2 Lowest()
    {
        // E1M2 has no NaN, all bit patterns are valid
        // 1 1 11 = sign=1, exp=1, man=3 -> -2^0 * 1.75 = -1.75
        return Float4E1M2(LOWEST_VALUE, FromBits()); // 1 1 11 = -1.75
    }

    static constexpr Float4E1M2 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 1 00)
        return Float4E1M2(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float4E1M2ToFloat(Float4E1M2 fp4);
    static Float4E1M2 FloatToFloat4E1M2(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float4E1M2& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float4E1M2& a [[maybe_unused]])
{
    return false; // E1M2 has no infinity
}

inline bool isnan(const op::Float4E1M2& a) { return a.IsNaN(); }

inline bool isfinite(const op::Float4E1M2& a) { return !a.IsNaN(); }

template <>
class numeric_limits<op::Float4E1M2> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false; // E1M2 has no NaN
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
    static constexpr int min_exponent = 0; // 2^0 for denorm (man/4)
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0; // 2^0 for normal
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        return 0.25f; // 2^0 * (1/4) = 0.25 (man=1, exp=0)
    }

    static constexpr op::Float4E1M2 min() { return op::Float4E1M2::MinPositiveNormal(); }
    static constexpr op::Float4E1M2 lowest() { return op::Float4E1M2::Lowest(); }
    static constexpr op::Float4E1M2 max() { return op::Float4E1M2::Highest(); }
    static constexpr op::Float4E1M2 epsilon() { return op::Float4E1M2::Epsilon(); }
    static constexpr op::Float4E1M2 infinity()
    {
        return op::Float4E1M2::Highest(); // No infinity, use max
    }
    static constexpr op::Float4E1M2 quiet_NaN()
    {
        // E1M2 has no NaN, return zero as fallback
        return op::Float4E1M2(0x00, op::Float4E1M2::FromBits());
    }
    static constexpr op::Float4E1M2 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT4_E1M2_H
