/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include "gtest/gtest.h"

#include "float_benchmark_data.h"
#include "opdev/hifloat4.h"

namespace op {
namespace benchmark {

// ============================================================================
// HiFloat4 标杆值定义
// HiFloat4: 1 sign + 1 exp (bias=1) + 2 mantissa, max=1.75
// 格式与 Float4E1M2 相同，专为神经网络量化优化
// ============================================================================

// 所有可表示的正数值测试用例
static constexpr BenchmarkCase<HiFloat4> kHiFloat4_AllPositiveCases[] = {
    {0.0f, 0x0, 0.0f, 0.0f, "zero", "CANN"},
    {0.25f, 0x1, 0.25f, 0.0f, "denorm 0.25 (man=1)", "CANN"},
    {0.5f, 0x2, 0.5f, 0.0f, "denorm 0.5 (man=2)", "CANN"},
    {0.75f, 0x3, 0.75f, 0.0f, "denorm 0.75 (man=3)", "CANN"},
    {1.0f, 0x4, 1.0f, 0.0f, "1.0 (exp=1, man=0)", "CANN"},
    {1.25f, 0x5, 1.25f, 0.0f, "1.25 (exp=1, man=1)", "CANN"},
    {1.5f, 0x6, 1.5f, 0.0f, "1.5 (exp=1, man=2)", "CANN"},
    {1.75f, 0x7, 1.75f, 0.0f, "max 1.75 (exp=1, man=3)", "CANN"},
};

// ============================================================================
// 测试类
// ============================================================================

class HiFloat4Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<HiFloat4>& tc)
    {
        HiFloat4 result(tc.input);

        // 验证位表示
        EXPECT_EQ(result.value, tc.expected_bits) << "Input: " << tc.input << ", Desc: " << tc.description;

        // 验证转换值
        float actual = static_cast<float>(result);
        EXPECT_NEAR(actual, tc.expected_value, tc.tolerance) << "Desc: " << tc.description;
    }
};

// ============================================================================
// 测试用例
// ============================================================================

TEST_F(HiFloat4Benchmark, AllPositiveValues)
{
    for (const auto& tc : kHiFloat4_AllPositiveCases) {
        RunBenchmark(tc);
    }
}

TEST_F(HiFloat4Benchmark, NegativeValues)
{
    // 负值：最高位为符号位
    HiFloat4 neg_quarter(-0.25f);
    EXPECT_EQ(neg_quarter.value, 0x9); // 1 0 01
    EXPECT_FLOAT_EQ(static_cast<float>(neg_quarter), -0.25f);

    HiFloat4 neg_one(-1.0f);
    EXPECT_EQ(neg_one.value, 0xC); // 1 1 00
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    HiFloat4 neg_max(-1.75f);
    EXPECT_EQ(neg_max.value, 0xF); // 1 1 11
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -1.75f);
}

TEST_F(HiFloat4Benchmark, MaxValue)
{
    HiFloat4 max_val(1.75f);
    float result = static_cast<float>(max_val);
    EXPECT_FLOAT_EQ(result, 1.75f);
    EXPECT_EQ(max_val.value, 0x7);
}

TEST_F(HiFloat4Benchmark, OverflowClamp)
{
    HiFloat4 overflow(100.0f);
    float result = static_cast<float>(overflow);
    // 溢出应该 clamp 到最大值 1.75
    EXPECT_FLOAT_EQ(result, 1.75f);
    EXPECT_EQ(overflow.value, 0x7);
}

TEST_F(HiFloat4Benchmark, UnderflowToZero)
{
    HiFloat4 tiny(0.05f);
    float result = static_cast<float>(tiny);
    // 小于最小非正规值应舍入到 0 或 0.25
    EXPECT_TRUE(result == 0.0f || result == 0.25f);
}

TEST_F(HiFloat4Benchmark, NaNClampToMax)
{
    // HiFloat4 没有 NaN，输入 NaN 应 clamp 到最大值
    HiFloat4 nan_val(std::nanf(""));
    float result = static_cast<float>(nan_val);
    EXPECT_FLOAT_EQ(result, 1.75f); // 正 NaN -> +max
}

TEST_F(HiFloat4Benchmark, InfinityClampToMax)
{
    // HiFloat4 没有无穷大，输入 Inf 应 clamp 到最大值
    HiFloat4 inf_val(std::numeric_limits<float>::infinity());
    float result = static_cast<float>(inf_val);
    EXPECT_FLOAT_EQ(result, 1.75f);

    HiFloat4 neg_inf_val(-std::numeric_limits<float>::infinity());
    result = static_cast<float>(neg_inf_val);
    EXPECT_FLOAT_EQ(result, -1.75f);
}

TEST_F(HiFloat4Benchmark, RoundingBehavior)
{
    // 测试舍入行为 (round-to-nearest-even)
    // 0.625 应该舍入到 0.5 或 0.75
    HiFloat4 val1(0.625f);
    float r1 = static_cast<float>(val1);
    EXPECT_TRUE(r1 == 0.5f || r1 == 0.75f);

    // 1.125 应该舍入到 1.0 或 1.25
    HiFloat4 val2(1.125f);
    float r2 = static_cast<float>(val2);
    EXPECT_TRUE(r2 == 1.0f || r2 == 1.25f);
}

TEST_F(HiFloat4Benchmark, DenormRange)
{
    // 验证非正规值范围 [0.25, 0.75]
    HiFloat4 d1(0.25f);
    EXPECT_FLOAT_EQ(static_cast<float>(d1), 0.25f);

    HiFloat4 d2(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(d2), 0.5f);

    HiFloat4 d3(0.75f);
    EXPECT_FLOAT_EQ(static_cast<float>(d3), 0.75f);
}

TEST_F(HiFloat4Benchmark, NormalRange)
{
    // 验证正规值范围 [1.0, 1.75]
    HiFloat4 n1(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(n1), 1.0f);

    HiFloat4 n2(1.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(n2), 1.5f);

    HiFloat4 n3(1.75f);
    EXPECT_FLOAT_EQ(static_cast<float>(n3), 1.75f);
}

TEST_F(HiFloat4Benchmark, StaticConstants)
{
    // 验证静态常量
    EXPECT_EQ(HiFloat4::Highest().value, 0x7);
    EXPECT_EQ(HiFloat4::Lowest().value, 0xF);
    EXPECT_EQ(HiFloat4::MinPositiveNormal().value, 0x4);
    EXPECT_EQ(HiFloat4::Epsilon().value, 0x1);
}

TEST_F(HiFloat4Benchmark, ZeroSignHandling)
{
    // 验证 +0 和 -0
    HiFloat4 pos_zero(0.0f);
    EXPECT_EQ(pos_zero.value, 0x0);
    EXPECT_TRUE(pos_zero.IsZero());

    HiFloat4 neg_zero(-0.0f);
    EXPECT_EQ(neg_zero.value, 0x8); // 符号位为 1
    EXPECT_TRUE(neg_zero.IsZero());
}

} // namespace benchmark
} // namespace op
