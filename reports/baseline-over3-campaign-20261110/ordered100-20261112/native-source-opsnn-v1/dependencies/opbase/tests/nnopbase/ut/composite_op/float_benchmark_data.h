/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOAT_BENCHMARK_DATA_H
#define FLOAT_BENCHMARK_DATA_H

#include <array>
#include <cmath>
#include <cstdint>

namespace op {
namespace benchmark {

// 测试类别
enum class TestCategory {
    BOUNDARY, // 边界值
    SPECIAL,  // 特殊值 (0, NaN, Inf)
    TYPICAL,  // 典型值
    OVERFLOW, // 溢出处理
    ROUNDING  // 舍入验证
};

// 单个测试用例
template <typename FloatType>
struct BenchmarkCase {
    float input;             // 输入值 (float32)
    uint8_t expected_bits;   // 期望的位表示
    float expected_value;    // 期望的转换回 float 的值
    float tolerance;         // 允许误差
    const char* description; // 用例描述
    const char* source;      // 来源标识
};

// 辅助函数：检查 NaN
inline bool IsNanOrBothNan(float expected, float actual)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    return false;
}

} // namespace benchmark
} // namespace op

#endif // FLOAT_BENCHMARK_DATA_H
