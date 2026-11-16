/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp16.h
 * \brief Half precision float
 */
#ifndef OP_COMMON_OP_HOST_UTIL_FP16_H
#define OP_COMMON_OP_HOST_UTIL_FP16_H

#include <stdint.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include "opbase_export.h"

namespace Ops {
namespace Base {
/**
 * @ingroup fp16_t
 * @brief   Half precision float
 *         bit15:       1 bit SIGN      +---+-----+------------+
 *         bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *         bit0-9:      10bit MAN       +---+-----+------------+
 *
 */
struct OPBASE_API tagFp16 {
    uint16_t val;

public:
    /**
     * @ingroup fp16_t constructor
     * @brief   Constructor without any param(default constructor)
     */
    tagFp16(void) : val(0x0u) {}
    /**
     * @ingroup all type constructor
     * @brief   Constructor with all type
     */
    template <typename T>
    tagFp16(const T& value)
    {
        *this = value;
    }
    /**
     * @ingroup fp16_t constructor
     * @brief   Constructor with an uint16_t value
     */
    tagFp16(const uint16_t& uiVal) : val(uiVal) {}
    /**
     * @ingroup fp16_t constructor
     * @brief   Constructor with a fp16_t object(copy constructor)
     */
    tagFp16(const tagFp16& fp) : val(fp.val) {}

    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be added
     * @brief   Override addition operator to performing fp16_t addition
     * @return  Return fp16_t result of adding this and fp
     */
    tagFp16 operator+(const tagFp16 fp) const;
    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be subtracted
     * @brief   Override addition operator to performing fp16_t subtraction
     * @return  Return fp16_t result of subtraction fp from this
     */
    tagFp16 operator-(const tagFp16 fp) const;
    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be multiplied
     * @brief   Override multiplication operator to performing fp16_t multiplication
     * @return  Return fp16_t result of multiplying this and fp
     */
    tagFp16 operator*(const tagFp16 fp) const;
    /**
     * @ingroup fp16_t math operator divided
     * @param [in] fp fp16_t object to be divided
     * @brief   Override division operator to performing fp16_t division
     * @return  Return fp16_t result of division this by fp
     */
    tagFp16 operator/(const tagFp16 fp) const;
    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be added
     * @brief   Override addition operator to performing fp16_t addition
     * @return  Return fp16_t result of adding this and fp
     */
    tagFp16 operator+=(const tagFp16 fp);
    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be subtracted
     * @brief   Override addition operator to performing fp16_t subtraction
     * @return  Return fp16_t result of subtraction fp from this
     */
    tagFp16 operator-=(const tagFp16 fp);
    /**
     * @ingroup fp16_t math operator
     * @param [in] fp fp16_t object to be multiplied
     * @brief   Override multiplication operator to performing fp16_t multiplication
     * @return  Return fp16_t result of multiplying this and fp
     */
    tagFp16 operator*=(const tagFp16 fp);
    /**
     * @ingroup fp16_t math operator divided
     * @param [in] fp fp16_t object to be divided
     * @brief   Override division operator to performing fp16_t division
     * @return  Return fp16_t result of division this by fp
     */
    tagFp16 operator/=(const tagFp16 fp);

    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t if-equal comparison
     * @return  Return boolean result of if-equal comparison of this and fp.
     */
    bool operator==(const tagFp16& fp) const;
    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t not-equal comparison
     * @return  Return boolean result of not-equal comparison of this and fp.
     */
    bool operator!=(const tagFp16& fp) const;
    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t greater-than comparison
     * @return  Return boolean result of greater-than comparison of this and fp.
     */
    bool operator>(const tagFp16& fp) const;
    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t greater-equal comparison
     * @return  Return boolean result of greater-equal comparison of this and fp.
     */
    bool operator>=(const tagFp16& fp) const;
    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t less-than comparison
     * @return  Return boolean result of less-than comparison of this and fp.
     */
    bool operator<(const tagFp16& fp) const;
    /**
     * @ingroup fp16_t math compare operator
     * @param [in] fp fp16_t object to be compared
     * @brief   Override basic comparison operator to performing fp16_t less-equal comparison
     * @return  Return boolean result of less-equal comparison of this and fp.
     */
    bool operator<=(const tagFp16& fp) const;

    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] fp fp16_t object to be copy to fp16_t
     * @brief   Override basic evaluation operator to copy fp16_t to a new fp16_t
     * @return  Return fp16_t result from fp
     */
    tagFp16& operator=(const tagFp16& fp);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] fVal float object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert float to fp16_t
     * @return  Return fp16_t result from fVal
     */
    tagFp16& operator=(const float& fVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] dVal double object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert double to fp16_t
     * @return  Return fp16_t result from dVal
     */
    tagFp16& operator=(const double& dVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] iVal float object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert float to fp16_t
     * @return  Return fp16_t result from iVal
     */
    tagFp16& operator=(const int8_t& iVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] uiVal uint8_t object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert uint8_t to fp16_t
     * @return  Return fp16_t result from uiVal
     */
    tagFp16& operator=(const uint8_t& uiVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] iVal int16_t object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert int16_t to fp16_t
     * @return  Return fp16_t result from iVal
     */
    tagFp16& operator=(const int16_t& iVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] uiVal uint16_t object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert uint16_t to fp16_t
     * @return  Return fp16_t result from uiVal
     */
    tagFp16& operator=(const uint16_t& uiVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] iVal int32_t object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert int32_t to fp16_t
     * @return  Return fp16_t result from iVal
     */
    tagFp16& operator=(const int32_t& iVal);
    /**
     * @ingroup fp16_t math evaluation operator
     * @param [in] uiVal uint32_t object to be converted to fp16_t
     * @brief   Override basic evaluation operator to convert uint32_t to fp16_t
     * @return  Return fp16_t result from uiVal
     */
    tagFp16& operator=(const uint32_t& uiVal);
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to float/fp32
     * @return  Return float/fp32 value of fp16_t
     */
    operator float() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to double/fp64
     * @return  Return double/fp64 value of fp16_t
     */
    operator double() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to int8_t
     * @return  Return int8_t value of fp16_t
     */
    operator int8_t() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to uint8_t
     * @return  Return uint8_t value of fp16_t
     */
    operator uint8_t() const;
    /**
     * @ingroup fp16_t conversion
     * @brief   Override convert operator to convert fp16_t to int16_t
     * @return  Return int16_t value of fp16_t
     */
    operator int16_t() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to uint16_t
     * @return  Return uint16_t value of fp16_t
     */
    operator uint16_t() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to int32_t
     * @return  Return int32_t value of fp16_t
     */
    operator int32_t() const;
    /**
     * @ingroup fp16_t math conversion
     * @brief   Override convert operator to convert fp16_t to int64_t
     * @return  Return int64_t value of fp16_t
     */
    operator uint32_t() const;
    /**
     * @ingroup fp16_t judgment method
     * @param [in] fp fp16_t object to be judgement
     * @brief   whether a fp16_t is inifinite
     * @return  Returns 1:+INF -1:-INF 0:not INF
     */
    int IsInf() const;

    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to float/fp32
     * @return  Return float/fp32 value of fp16_t
     */
    float toFloat();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to double/fp64
     * @return  Return double/fp64 value of fp16_t
     */
    double toDouble();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to int8_t
     * @return  Return int8_t value of fp16_t
     */
    int8_t toInt8();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to uint8_t
     * @return  Return uint8_t value of fp16_t
     */
    uint8_t toUInt8();
    /**
     * @ingroup fp16_t conversion
     * @brief   Convert fp16_t to int16_t
     * @return  Return int16_t value of fp16_t
     */
    int16_t toInt16();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to uint16_t
     * @return  Return uint16_t value of fp16_t
     */
    uint16_t toUInt16();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to int32_t
     * @return  Return int32_t value of fp16_t
     */
    int32_t toInt32();
    /**
     * @ingroup fp16_t math conversion
     * @brief   Convert fp16_t to int64_t
     * @return  Return int64_t value of fp16_t
     */
    uint32_t toUInt32();
};
using fp16_t = tagFp16;
} // namespace Base
}; // namespace Ops

#endif /*OP_COMMON_OP_HOST_UTIL_FP16_H*/
