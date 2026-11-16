/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_UTILS_KERNEL_UTIL_H_
#define AICPU_UTILS_KERNEL_UTIL_H_

#include <climits>
#include <cmath>
#include <sstream>

#include "cpu_context.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
constexpr uint32_t kResvCpuNum = 2;
constexpr uint32_t kThreadNum = 32;
constexpr uint32_t kFirstInputIndex = 0;
constexpr uint32_t kSecondInputIndex = 1;
constexpr uint32_t kThirdInputIndex = 2;
constexpr uint32_t kFourthInputIndex = 3;
constexpr uint32_t kFifthInputIndex = 4;
constexpr uint32_t kSixthInputIndex = 5;
constexpr uint32_t kSeventhInputIndex = 6;
constexpr uint32_t kEighthInputIndex = 7;
constexpr uint32_t kNinthInputIndex = 8;
constexpr uint32_t kTenthInputIndex = 9;
constexpr uint32_t kEleventhInputIndex = 10;
constexpr uint32_t kTwelfthInputIndex = 11;
constexpr uint32_t kFirstOutputIndex = 0;
constexpr uint32_t kSecondOutputIndex = 1;
constexpr uint32_t kThirdOutputIndex = 2;
constexpr int32_t kDynamicInput = -1;
constexpr int32_t kDynamicOutput = -2;
constexpr uint64_t kEigenAlignmentBytes = 16;

constexpr uint64_t kFormatNCHWIndexN = 0;
constexpr uint64_t kFormatNCHWIndexC = 1;
constexpr uint64_t kFormatNCHWIndexH = 2;
constexpr uint64_t kFormatNCHWIndexW = 3;

constexpr uint64_t kFormatCHWIndexC = 0;
constexpr uint64_t kFormatCHWIndexH = 1;
constexpr uint64_t kFormatCHWIndexW = 2;

constexpr uint64_t kFormatNHWCIndexN = 0;
constexpr uint64_t kFormatNHWCIndexH = 1;
constexpr uint64_t kFormatNHWCIndexW = 2;
constexpr uint64_t kFormatNHWCIndexC = 3;

constexpr uint64_t kFormatHWCIndexH = 0;
constexpr uint64_t kFormatHWCIndexW = 1;
constexpr uint64_t kFormatHWCIndexC = 2;

const size_t INPUT_NUM0 = 0;
const size_t INPUT_NUM1 = 1;
const size_t INPUT_NUM2 = 2;
const size_t INPUT_NUM3 = 3;
const size_t INPUT_NUM4 = 4;
const size_t INPUT_NUM5 = 5;
const size_t INPUT_NUM6 = 6;
const size_t INPUT_NUM7 = 7;
const size_t INPUT_NUM8 = 8;
const size_t INPUT_NUM9 = 9;
const size_t INPUT_NUM32 = 32;

constexpr int kRank1 = 1;
constexpr int kRank2 = 2;
constexpr int kRank3 = 3;
constexpr int kRank4 = 4;
constexpr int kRank5 = 5;
constexpr int kRank6 = 6;
constexpr int kRank7 = 7;
constexpr int kRank8 = 8;

constexpr int kShift8 = 8;
constexpr int kShift32 = 32;

enum class SocVersion {
    ASCEND910 = 0,
    ASCEND910B,
    ASCEND910_93,
    ASCEND950,
    ASCEND910E,
    ASCEND310,
    ASCEND310P,
    ASCEND310B,
    ASCEND310C,
    ASCEND610LITE,
    RESERVED_VERSION = 99999
};
/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string ConcatString(T arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(arg_left...);
    return oss.str();
}

/**
 * @brief get debug string of vector
 * @param values values in vector
 * @return string of values
 */
template <typename T>
inline std::string VectorToString(const std::vector<T>& values)
{
    std::stringstream ss;
    for (auto iter = values.begin(); iter != values.end(); ++iter) {
        ss << *iter;
        if (iter != values.end() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

template <typename T>
std::string FmtToStr(const T& t)
{
    std::string fmt;
    std::stringstream st;
    st << "[" << t << "]";
    fmt = st.str();
    return fmt;
}

std::string FormatToSerialString(Format format);

/**
 * Get primary-format from format,
 * in bits field:
 * ------------------------------------------
 * |  1 byte  |   2 bytes  |     1 byt      |
 * |----------|------------|----------------|
 * | reserved | sub-format | primary-format |
 * ------------------------------------------
 * @param format
 * @return
 */
inline int32_t GetPrimaryFormat(int32_t format) { return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xff); }

inline int32_t GetSubFormat(int32_t format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00) >> kShift8);
}

inline bool HasSubFormat(int32_t format) { return GetSubFormat(format) > 0; }

/**
 * @brief Judge whether tensor is empty
 * @param tensor need judged tensor
 * @return true: is empty tensor, false: isn't empty tensor
 */
bool IsEmptyTensor(Tensor* tensor);

/**
 * @brief multiply two nonnegative int64's
 * @param x mul value x
 * @param y mul value y
 * @param xy product of x and y
 * @return true: normal, false: overflow
 */
inline bool MulWithoutOverflow(const int64_t x, const int64_t y, int64_t& xy)
{
    // Multiply in uint64 rather than int64 since signed overflow is undefined.
    // Negative values will wrap around to large unsigned values in the casts
    // (see section 4.7 [conv.integral] of the C++14 standard).
    const uint64_t ux = static_cast<uint64_t>(x);
    const uint64_t uy = static_cast<uint64_t>(y);
    const uint64_t uxy = ux * uy;

    // Check if we overflow uint64, using a cheap check if both inputs are small
    if (((ux | uy) >> kShift32) != 0) {
        // Ensure nonnegativity.  Note that negative numbers will appear "large"
        // to the unsigned comparisons above.
        if (x < 0 || y < 0) {
            KERNEL_LOG_ERROR("Can't multiply negative numbers.");
            return false;
        }

        // Otherwise, detect overflow using a division
        if (ux != 0 && uxy / ux != uy) {
            return false;
        }
    }

    // Cast back to signed.  Any negative value will signal an error.
    xy = static_cast<int64_t>(uxy);
    return true;
}

/**
 * @brief add two int64's
 * @param x add value x
 * @param y add value y
 * @param sum sum of x and y
 * @return true: normal, false: overflow
 */
inline bool AddWithoutOverflow(const int64_t x, const int64_t y, int64_t& sum)
{
    const uint64_t ux = static_cast<uint64_t>(x);
    const uint64_t uy = static_cast<uint64_t>(y);
    const uint64_t usum = ux + uy;
    sum = static_cast<int64_t>(usum);

    return !(((x >= 0) == (y >= 0)) && ((sum >= 0) != (x >= 0)));
}

/**
 * @brief check whether uint32 multiplication can result in overflow
 * @param [in] a  multiplicator
 * @param [in] b  multiplicator
 * @return bool
 */
inline bool CheckUint32MulOverflow(const uint32_t a, const uint32_t b)
{
    if ((a == 0U) || (b == 0U)) {
        return true;
    }

    if (a > (UINT32_MAX / b)) {
        return false;
    }

    return true;
}

/**
 * @brief check whether int32 multiplication can result in overflow
 * @param [in] a  multiplicator
 * @param [in] b  multiplicator
 * @return bool
 */
inline bool CheckInt32MulOverflow(const int32_t a, const int32_t b)
{
    if (a > 0) {
        if (b > 0) {
            if (a > (INT32_MAX / b)) {
                return false;
            }
        } else {
            if (b < (INT32_MIN / a)) {
                return false;
            }
        }
    } else {
        if (b > 0) {
            if (a < (INT32_MIN / b)) {
                return false;
            }
        } else {
            if ((a != 0) && (b < (INT32_MAX / a))) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief check whether uint64 multiplication can result in overflow
 * @param [in] a  multiplicator
 * @param [in] b  multiplicator
 * @return bool
 */
inline bool CheckUint64MulOverflow(const uint64_t a, const uint64_t b)
{
    if ((a == 0UL) || (b == 0UL)) {
        return true;
    }

    if (a > (UINT64_MAX / b)) {
        return false;
    }

    return true;
}

/**
 * @brief check whether int64 multiplication can result in overflow
 * @param [in] a  multiplicator
 * @param [in] b  multiplicator
 * @return bool
 */
inline bool CheckInt64MulOverflow(const int64_t a, const int64_t b)
{
    if (a > 0) {
        if (b > 0) {
            if (a > (INT64_MAX / b)) {
                return false;
            }
        } else {
            if (b < (INT64_MIN / a)) {
                return false;
            }
        }
    } else {
        if (b > 0) {
            if (a < (INT64_MIN / b)) {
                return false;
            }
        } else {
            if ((a != 0) && (b < (INT64_MAX / a))) {
                return false;
            }
        }
    }
    return true;
}

template <typename TI, typename TO>
inline auto PtrToPtr(TI* const ptr) -> TO*
{
    return reinterpret_cast<TO*>(ptr);
}

/**
 * @ingroup math_util
 * @brief check whether float addition can result in overflow
 *  @param [in] a  addend
 *  @param [in] b  addend
 * @return Status
 */
inline bool CheckFloatAddOverflow(float a, float b)
{
    if (!std::isfinite(static_cast<float>(a) + static_cast<float>(b))) {
        return false;
    }
    return true;
}

/**
 * @brief normal check for calculation
 * @param ctx context
 * @return status code
 */
uint32_t NormalMathCheck(CpuKernelContext& ctx);

/**
 * @brief normal check for kernel
 * @param ctx context
 * @param inputs_num num of inputs
 * @param outputs_num num of outputs
 * @return status code
 */
uint32_t NormalCheck(CpuKernelContext& ctx, const uint32_t inputs_num, const uint32_t outputs_num);

/**
 * @brief normal check for kernel
 * @param ctx context
 * @param inputs_num num of inputs
 * @param outputs_num num of outputs
 * @param attr_names names of attrs
 * @return status code
 */
uint32_t NormalCheck(CpuKernelContext& ctx, const uint32_t inputs_num, const uint32_t outputs_num,
                     const std::vector<std::string>& attr_names);

bool IsScalar(const std::vector<int64_t>& shape);

bool IsMatrix(const std::vector<int64_t>& shape);

bool IsVector(const std::vector<int64_t>& shape);

bool IsSquareMatrix(const std::vector<int64_t>& shape);
/**
 * @brief check if addr is aligned
 * @param addr address for check
 * @return true: aligned, false: not aligned
 */
bool AddrAlignedCheck(const void* addr, uint64_t alignment = kEigenAlignmentBytes);

bool IsVectorOrHigher(const std::vector<int64_t>& shape);

/**
 * @brief get data type from string
 * @param dtype_str string of data type
 * @return DataType
 */
DataType DType(std::string dtype_str);

/**
 * @brief get string from data type
 * @param dtype data type
 * @return string of data type
 */
std::string DTypeStr(DataType dtype);

bool BiggerMemCpy(void* dst_addr, const std::size_t dst_len, const void* src_addr, const std::size_t src_len);

bool BiggerMemSet(void* dst_addr, const std::size_t dst_len, const int c, const std::size_t count);

int64_t CeilMultiple(const int64_t x, const int64_t base);

template <typename T, typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
inline bool IsNanVal(T val)
{
    (void)val;
    return false;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
inline bool IsNanVal(T val)
{
    return std::isnan(val);
}

template <typename T, typename std::enable_if<std::is_same<T, Eigen::half>::value, T>::type* = nullptr>
inline bool IsNanVal(T val)
{
    return std::isnan(static_cast<float>(val));
}

template <typename T, typename std::enable_if<std::is_same<T, std::complex<float>>::value, T>::type* = nullptr>
inline bool IsNanVal(T val)
{
    return std::isnan(val.real()) || std::isnan(val.imag());
}

template <typename T, typename std::enable_if<std::is_same<T, std::complex<double>>::value, T>::type* = nullptr>
inline bool IsNanVal(T val)
{
    return std::isnan(val.real()) || std::isnan(val.imag());
}

int64_t GetMaxCompilerCoreNum();

uint64_t CalFileSize(const std::string& file_path);

uint64_t AlignedTo(const uint64_t size, const uint64_t alignment);

template <typename T>
typename std::enable_if<!(std::is_same<T, float>::value || std::is_same<T, double>::value ||
                          std::is_same<T, Eigen::half>::value),
                        bool>::type
IsValueEqual(T a, T b)
{
    return a == b;
}

template <typename T>
typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, bool>::type IsValueEqual(T a,
                                                                                                                  T b)
{
    constexpr T epsilon = std::is_same<T, float>::value ? 1e-5f : 1e-9;
    return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
}

template <typename T>
typename std::enable_if<std::is_same<T, Eigen::half>::value, bool>::type IsValueEqual(T a, T b)
{
    constexpr float epsilon = 1e-3f;
    return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
}
} // namespace aicpu
#endif
