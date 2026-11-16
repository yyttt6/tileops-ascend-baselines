/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_HIFP8_H
#define OP_API_HIFP8_H

#include <cstdint>
#include <iostream>
#include <limits>

namespace op {

// HiFloat8 (HiF8) is an 8-bit floating point format developed by Huawei.
// It uses a tapered precision encoding with variable-length prefix codes.
//
// Format overview:
// - Normal mode: standard binary scientific notation
// - Denormal mode: extended exponent range for small values
// - Exponent range: [-22, 15] (38 exponent values total)
// - Mantissa width varies: 0-3 bits depending on exponent magnitude

struct HiFloat8 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits() { return FromBitsTag(); }

    uint8_t value;

    static constexpr uint8_t HIF8_ZERO_VALUE = 0;         // Zero representation
    static constexpr uint8_t HIF8_NAN_VALUE = 0b10000000; // NaN representation
    static constexpr uint8_t HIF8_INF_VALUE = 0b01101111; // INF representation

    // Default constructor yields zero value
    constexpr HiFloat8() : value(HIF8_ZERO_VALUE) {}

    constexpr HiFloat8(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits) {}

    HiFloat8(float f32);

    // Convert to float
    operator float() const;

    // Check if the value is NaN
    bool IsNaN() const;
    // Check if the value is Infinity
    bool IsInf() const;
    // Check if the value is zero
    bool IsZero() const;

    // Convert FP32 bits to HiF8 bits
    static uint8_t BitsFromFp32(uint32_t f32);

private:
    static float Hifp8ToFloat(HiFloat8 hif8);
};

// Stream output operator
inline std::ostream& operator<<(std::ostream& os, const HiFloat8& dt)
{
    os << static_cast<float>(dt);
    return os;
}

static_assert(sizeof(HiFloat8) == sizeof(uint8_t), "sizeof HiFloat8 must be 1");

} // namespace op

namespace std {

inline bool isinf(const op::HiFloat8& a) { return a.IsInf(); }

inline bool isnan(const op::HiFloat8& a) { return a.IsNaN(); }

inline bool isfinite(const op::HiFloat8& a) { return !a.IsInf() && !a.IsNaN(); }

template <>
class numeric_limits<op::HiFloat8> {
public:
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 3; // Maximum mantissa bits
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int min_exponent = -22; // Minimum exponent
    static constexpr int min_exponent10 = -6;
    static constexpr int max_exponent = 15; // Maximum exponent
    static constexpr int max_exponent10 = 4;
    static constexpr int radix = 2;

    static constexpr op::HiFloat8 min()
    {
        return op::HiFloat8(0b00001000, op::HiFloat8::FromBits()); // Smallest positive normal
    }

    static constexpr op::HiFloat8 lowest()
    {
        return op::HiFloat8(0b11111111, op::HiFloat8::FromBits()); // Most negative value
    }

    static constexpr op::HiFloat8 max()
    {
        return op::HiFloat8(0b01101111, op::HiFloat8::FromBits()); // Largest positive value (infinity)
    }

    static constexpr op::HiFloat8 epsilon()
    {
        return op::HiFloat8(0b00000001, op::HiFloat8::FromBits()); // Machine epsilon
    }

    static constexpr op::HiFloat8 round_error()
    {
        return op::HiFloat8(0b00011000, op::HiFloat8::FromBits()); // 0.5f in HiF8 format
    }

    static constexpr op::HiFloat8 infinity()
    {
        return op::HiFloat8(op::HiFloat8::HIF8_INF_VALUE, op::HiFloat8::FromBits());
    }

    static constexpr op::HiFloat8 quiet_NaN()
    {
        return op::HiFloat8(op::HiFloat8::HIF8_NAN_VALUE, op::HiFloat8::FromBits());
    }

    static constexpr op::HiFloat8 signaling_NaN()
    {
        return op::HiFloat8(op::HiFloat8::HIF8_NAN_VALUE, op::HiFloat8::FromBits());
    }

    static constexpr op::HiFloat8 denorm_min()
    {
        return op::HiFloat8(0b00000001, op::HiFloat8::FromBits()); // Smallest positive denormal
    }
};

} // namespace std

#endif // OP_API_HIFP8_H
