/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_HIFLOAT4_H
#define OP_API_HIFLOAT4_H

#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief HiFloat4 floating-point format
 *
 * HiFloat4 is a 4-bit floating-point format designed for high-efficiency
 * neural network computations. It provides an optimal balance between
 * dynamic range and precision for quantized inference and training scenarios.
 *
 * Format: 1 sign bit, 1 exponent bit, 2 mantissa bits
 *         +---+--+---+
 *         | S |E |MM |
 *         +---+--+---+
 *
 * Key characteristics:
 * - Bias: 1
 * - Max normal value: 1.75 (0x7 = 0 1 11, exp=1, man=3 -> 2^0 * 1.75)
 * - Min normal value: -1.75 (0xF = 1 1 11, exp=1, man=3 -> -2^0 * 1.75)
 * - Min positive normal: 2^0 = 1.0 (0x4 = 0 1 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 2^0 * 0.25 = 0.25 (0x1 = 0 0 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 *
 * This format is particularly suitable for:
 * - Quantized neural network inference
 * - Low-precision training with gradient scaling
 * - Memory-bandwidth constrained applications
 */
struct HiFloat4 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    // HiFloat4 format semantic constants
    static constexpr uint8_t HIGHEST_VALUE = 0x07;        // Max value 1.75
    static constexpr uint8_t LOWEST_VALUE = 0x0F;         // Min value -1.75
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04; // Min positive normal 1.0
    static constexpr uint8_t BITS_MASK = 0x0F;            // 4-bit effective mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;        // Epsilon value

    // Default constructor - initialize to zero
    constexpr HiFloat4() : value(0) {}

    constexpr HiFloat4(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK) {}

    HiFloat4(float v);

    operator float() const;
    operator double() const;

    static constexpr HiFloat4 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return HiFloat4(EPSILON_VALUE, FromBits()); // 0 0 01 = denorm 0.25
    }

    static constexpr HiFloat4 Highest()
    {
        // exp=1, man=3 -> 2^0 * 1.75 = 1.75 (0 1 11 = 0x7)
        return HiFloat4(HIGHEST_VALUE, FromBits());
    }

    static constexpr HiFloat4 Lowest()
    {
        // 1 1 11 = sign=1, exp=1, man=3 -> -2^0 * 1.75 = -1.75
        return HiFloat4(LOWEST_VALUE, FromBits()); // 1 1 11 = -1.75
    }

    static constexpr HiFloat4 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 1 00)
        return HiFloat4(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const;
    bool IsNaN() const;
    bool IsInf() const;

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float HiFloat4ToFloat(HiFloat4 fp4);
    static HiFloat4 FloatToHiFloat4(float f);
};

inline std::ostream& operator<<(std::ostream& os, const HiFloat4& dt)
{
    os << static_cast<float>(dt);
    return os;
}

} // namespace op

namespace std {

inline bool isinf(const op::HiFloat4& a [[maybe_unused]])
{
    return false; // HiFloat4 has no infinity
}

inline bool isnan(const op::HiFloat4& a) { return a.IsNaN(); }

inline bool isfinite(const op::HiFloat4& a) { return !a.IsNaN(); }

template <>
class numeric_limits<op::HiFloat4> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
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
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error() { return 0.5f; }
    static constexpr float denorm_min()
    {
        return 0.25f; // 2^0 * (1/4) = 0.25 (man=1, exp=0)
    }

    static constexpr op::HiFloat4 min() { return op::HiFloat4::MinPositiveNormal(); }
    static constexpr op::HiFloat4 lowest() { return op::HiFloat4::Lowest(); }
    static constexpr op::HiFloat4 max() { return op::HiFloat4::Highest(); }
    static constexpr op::HiFloat4 epsilon() { return op::HiFloat4::Epsilon(); }
    static constexpr op::HiFloat4 infinity()
    {
        return op::HiFloat4::Highest(); // No infinity, use max
    }
    static constexpr op::HiFloat4 quiet_NaN()
    {
        // HiFloat4 has no NaN, return zero as fallback
        return op::HiFloat4(0x00, op::HiFloat4::FromBits());
    }
    static constexpr op::HiFloat4 signaling_NaN() { return quiet_NaN(); }
};

} // namespace std

#endif // OP_API_HIFLOAT4_H
