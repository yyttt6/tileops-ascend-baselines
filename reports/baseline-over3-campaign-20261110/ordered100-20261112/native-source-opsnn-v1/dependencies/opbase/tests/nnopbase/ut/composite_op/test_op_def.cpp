/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_def.h"
#include "gtest/gtest.h"

using namespace op;

class TestOpDef : public testing::Test {};

TEST_F(TestOpDef, OpImplModeToStr)
{
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_RESERVED), "unknown");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_HIGH_PRECISION), "high_precision");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_HIGH_PERFORMANCE), "high_performance");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_SUPER_PERFORMANCE), "super_performance");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX), "support_out_of_bound_index");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_ENABLE_FLOAT32_EXECUTION), "enable_float32_execution");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_ENABLE_HI_FLOAT32_EXECUTION), "enable_hi_float32_execution");
    EXPECT_EQ(op::ToString(OpImplMode::IMPL_MODE_DEFAULT), "default");
}

TEST_F(TestOpDef, StrToOpImplMode)
{
    EXPECT_EQ(op::ToOpImplMode("unknown"), OpImplMode::IMPL_MODE_RESERVED);
    EXPECT_EQ(op::ToOpImplMode("high_precision"), OpImplMode::IMPL_MODE_HIGH_PRECISION);
    EXPECT_EQ(op::ToOpImplMode("high_performance"), OpImplMode::IMPL_MODE_HIGH_PERFORMANCE);
    EXPECT_EQ(op::ToOpImplMode("super_performance"), OpImplMode::IMPL_MODE_SUPER_PERFORMANCE);
    EXPECT_EQ(op::ToOpImplMode("support_out_of_bound_index"), OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX);
    EXPECT_EQ(op::ToOpImplMode("enable_float32_execution"), OpImplMode::IMPL_MODE_ENABLE_FLOAT32_EXECUTION);
    EXPECT_EQ(op::ToOpImplMode("enable_hi_float32_execution"), OpImplMode::IMPL_MODE_ENABLE_HI_FLOAT32_EXECUTION);
    EXPECT_EQ(op::ToOpImplMode("default"), OpImplMode::IMPL_MODE_DEFAULT);
}
