/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "op_common/op_host/util/math_util.h"

using namespace Ops::Base;
TEST(TestMathUtil, testFloorDiv)
{
    EXPECT_EQ(FloorDiv<int32_t>(10, 3), 3);
    EXPECT_EQ(FloorDiv<int32_t>(87, 20), 4);
    EXPECT_EQ(FloorDiv<int32_t>(1500, 512), 2);
    EXPECT_EQ(FloorDiv<int32_t>(130, 64), 2);
    EXPECT_EQ(FloorDiv<int64_t>(999, 100), 9);
    EXPECT_NE(FloorDiv<int64_t>(10, 3), 0);
    EXPECT_EQ(FloorDiv<int32_t>(std::numeric_limits<int32_t>::max(), 8), 268435455);
    EXPECT_EQ(FloorDiv<int64_t>(std::numeric_limits<int64_t>::max(), 64), 144115188075855871);
}

TEST(TestMathUtil, testFloorAlign)
{
    EXPECT_EQ(FloorAlign<int32_t>(12345, 4096), 12288);
    EXPECT_EQ(FloorAlign<int32_t>(130, 32), 128);
    EXPECT_EQ(FloorAlign<int32_t>(1023, 64), 960);
    EXPECT_EQ(FloorAlign<int32_t>(87, 20), 80);
    EXPECT_EQ(FloorAlign<int32_t>(185, 60), 180);
    EXPECT_EQ(FloorAlign<int32_t>(17, 4), 16);
    int64_t numX = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 2;
    EXPECT_EQ(FloorAlign<int64_t>(numX, 4), 4294967292);
}

TEST(TestMathUtil, testCeilDiv)
{
    EXPECT_EQ(CeilDiv<int32_t>(5000, 4096), 2);
    EXPECT_EQ(CeilDiv<int32_t>(130, 32), 5);
    EXPECT_EQ(CeilDiv<int32_t>(2500, 1024), 3);
    EXPECT_EQ(CeilDiv<int32_t>(999, 100), 10);
    EXPECT_EQ(CeilDiv<int32_t>(87, 20), 5);
    EXPECT_EQ(CeilDiv<int32_t>(std::numeric_limits<int32_t>::max(), 8), 268435456);
    EXPECT_EQ(CeilDiv<int64_t>(std::numeric_limits<int64_t>::max(), 8), 1152921504606846976);
}

TEST(TestMathUtil, testCeilAlign)
{
    EXPECT_EQ(CeilAlign<int32_t>(262144, 64), 262144);
    EXPECT_EQ(CeilAlign<int32_t>(1000, 64), 1024);
    EXPECT_EQ(CeilAlign<int32_t>(257, 32), 288);
    EXPECT_EQ(CeilAlign<int32_t>(1023, 8), 1024);
    EXPECT_EQ(CeilAlign<int32_t>(12345, 4096), 16384);
    EXPECT_EQ(CeilAlign<int32_t>(100000, 512), 100352);
    EXPECT_EQ(CeilAlign<int32_t>(std::numeric_limits<int32_t>::max(), 8), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(CeilAlign<int64_t>(std::numeric_limits<int64_t>::max(), 8), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(CeilAlign<uint64_t>(std::numeric_limits<uint64_t>::max(), 8), std::numeric_limits<uint64_t>::max());
}