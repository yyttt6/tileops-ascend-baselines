/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E8M0_H
#define OP_API_FLOAT8_E8M0_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP8 E8M0 floating-point format (scale format)
 *
 * Format: 8 exponent bits, 0 mantissa bits (unsigned)
 *         +--------+
 *         | EEEEEEEE|
 *         +--------+
 *
 * This is a special format used for scaling factors in MX formats.
 * It represents only positive powers of 2: value = 2^(exponent - bias)
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 127
 * - Max value: 2^127 (0xFE, exponent=254)
 * - Min positive value: 2^-126 (0x01, exponent=1)
 * - NaN: 0xFF (exponent=255) - reserved for NaN
 * - Zero is represented by 0x00 (exponent=0)
 * - No infinity encoding (only NaN at 0xFF)
 */
struct Float8E8M0 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E8M0 format constants (unsigned, 8 exponent bits) =============
    static constexpr uint8_t NAN_VALUE = 0xFF;          // 255 (NaN only, no infinity encoding)
    static constexpr uint8_t ONE_VALUE = 0x7F;          // 127 (2^0 = 1.0)
    static constexpr uint8_t HIGHEST_VALUE = 0xFE;      // 254 (max value: 2^127)
    static constexpr uint8_t MIN_POSITIVE_VALUE = 0x01; // 1 (min positive: 2^-126)

    // Default constructor - initialize to zero
    constexpr Float8E8M0() : value(0) {}

    constexpr Float8E8M0(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits) {}

    Float8E8M0(float v);

    operator float() const;
    operator double() const;

    static constexpr Float8E8M0 Epsilon()
    {
        // epsilon = nextafter(1.0) - 1.0 = 2.0 - 1.0 = 1.0
        return Float8E8M0(ONE_VALUE, FromBits()); // 2^0 = 1.0
    }

    static constexpr Float8E8M0 Highest()
    {
        // 2^127 (exponent=254)
        return Float8E8M0(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float8E8M0 Lowest()
    {
        // E8M0 is unsigned, lowest positive is MinPositive
        return Float8E8M0::MinPositive();
    }

    static constexpr Float8E8M0 MinPositive()
    {
        // 2^-126 (exponent=1)
        return Float8E8M0(MIN_POSITIVE_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;
    // Get the exponent value (unbiased)
    int32_t GetExponent() const;

private:
    static float Float8E8M0ToFloat(Float8E8M0 fp8);
    static Float8E8M0 FloatToFloat8E8M0(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float8E8M0& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float8E8M0& a) { return a.IsInf(); }

inline bool isnan(const op::Float8E8M0& a) { return a.IsNaN(); }

inline bool isfinite(const op::Float8E8M0& a)
{
    // E8M0 has no infinity encoding, only NaN at 0xFF
    // Zero (0x00) is a valid finite value in this context
    return !a.IsNaN();
}

template <>
class numeric_limits<op::Float8E8M0> {
public:
    static constexpr bool has_infinity = false; // E8M0 has no infinity encoding
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = false; // E8M0 is unsigned
    static constexpr bool is_specialized = true;
    static constexpr int digits = 0; // No mantissa bits
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 0;
    static constexpr int min_exponent = -126;
    static constexpr int min_exponent10 = -38;
    static constexpr int max_exponent = 127;
    static constexpr int max_exponent10 = 38;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 1.0f; // No fractional precision
    }
    static constexpr float denorm_min()
    {
        return 0.0f; // No denormals
    }

    static constexpr op::Float8E8M0 min() { return op::Float8E8M0::MinPositive(); }
    static constexpr op::Float8E8M0 lowest()
    {
        return op::Float8E8M0::MinPositive(); // Unsigned, lowest = min
    }
    static constexpr op::Float8E8M0 max() { return op::Float8E8M0::Highest(); }
    static constexpr op::Float8E8M0 epsilon() { return op::Float8E8M0::Epsilon(); }
    static constexpr op::Float8E8M0 infinity()
    {
        // E8M0 has no infinity encoding, return max value
        return op::Float8E8M0::Highest();
    }
    static constexpr op::Float8E8M0 quiet_NaN()
    {
        return op::Float8E8M0(op::Float8E8M0::NAN_VALUE, op::Float8E8M0::FromBits());
    }
    static constexpr op::Float8E8M0 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT8_E8M0_H
