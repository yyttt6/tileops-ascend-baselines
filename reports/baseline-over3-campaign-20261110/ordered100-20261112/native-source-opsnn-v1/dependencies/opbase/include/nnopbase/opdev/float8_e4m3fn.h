/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E4M3FN_H
#define OP_API_FLOAT8_E4M3FN_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP8 E4M3FN floating-point format (OCP standard)
 *
 * Format: 1 sign bit, 4 exponent bits, 3 mantissa bits (FN = Finite Number)
 *         +---+----+---+
 *         | S | EEEE|MMM|
 *         +---+----+---+
 *
 * - Bias: 7
 * - Max normal value: 448 (0x7E)
 * - Min positive normal: 2^-6 = 0.015625 (0x08)
 * - Supports NaN (0x7F, 0xFF only)
 * - No infinity representation (max finite value is 448)
 */
struct Float8E4M3FN {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // ============= E4M3FN format constants =============
    static constexpr uint8_t NAN_VALUE = 0x7F;            // 0 1111 111
    static constexpr uint8_t HIGHEST_VALUE = 0x7E;        // 0 1111 110 (max finite: 448)
    static constexpr uint8_t LOWEST_VALUE = 0xFE;         // 1 1111 110 (min finite: -448)
    static constexpr uint8_t EPSILON_VALUE = 0x20;        // 0 0100 000 (epsilon at 1.0)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x08; // 0 0001 000 (min positive normal)

    // Default constructor - initialize to zero
    constexpr Float8E4M3FN() : value(0) {}

    constexpr Float8E4M3FN(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits) {}

    Float8E4M3FN(float v);

    operator float() const;
    operator double() const;

    static constexpr Float8E4M3FN Epsilon()
    {
        // 2^-3 = 0.125
        return Float8E4M3FN(EPSILON_VALUE, FromBits()); // 0 0100 000 = 1.0, so epsilon is one mantissa step at 1.0
    }

    static constexpr Float8E4M3FN Highest()
    {
        // 448.0 = 0x7E (0 1111 110)
        return Float8E4M3FN(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float8E4M3FN Lowest()
    {
        // -448.0 = 0xFE (1 1111 110)
        return Float8E4M3FN(LOWEST_VALUE, FromBits());
    }

    static constexpr Float8E4M3FN MinPositiveNormal()
    {
        // 2^-6 = 0.015625 = 0x08 (0 0001 000)
        return Float8E4M3FN(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float8E4M3FNToFloat(Float8E4M3FN fp8);
    static Float8E4M3FN FloatToFloat8E4M3FN(float f);
};

inline std::ostream& operator<<(std::ostream& os, const Float8E4M3FN& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::Float8E4M3FN& a [[maybe_unused]])
{
    return false; // E4M3FN has no infinity
}

inline bool isnan(const op::Float8E4M3FN& a) { return a.IsNaN(); }

inline bool isfinite(const op::Float8E4M3FN& a) { return !a.IsNaN(); }

template <>
class numeric_limits<op::Float8E4M3FN> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
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
    static constexpr int max_digits10 = 2;
    static constexpr int min_exponent = -6;
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 9;
    static constexpr int max_exponent10 = 2;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        return 0.001953125f; // 2^-9
    }

    static constexpr op::Float8E4M3FN min() { return op::Float8E4M3FN::MinPositiveNormal(); }
    static constexpr op::Float8E4M3FN lowest() { return op::Float8E4M3FN::Lowest(); }
    static constexpr op::Float8E4M3FN max() { return op::Float8E4M3FN::Highest(); }
    static constexpr op::Float8E4M3FN epsilon() { return op::Float8E4M3FN::Epsilon(); }
    static constexpr op::Float8E4M3FN infinity()
    {
        return op::Float8E4M3FN::Highest(); // No infinity, return max
    }
    static constexpr op::Float8E4M3FN quiet_NaN()
    {
        return op::Float8E4M3FN(op::Float8E4M3FN::NAN_VALUE, op::Float8E4M3FN::FromBits());
    }
    static constexpr op::Float8E4M3FN signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_FLOAT8_E4M3FN_H
