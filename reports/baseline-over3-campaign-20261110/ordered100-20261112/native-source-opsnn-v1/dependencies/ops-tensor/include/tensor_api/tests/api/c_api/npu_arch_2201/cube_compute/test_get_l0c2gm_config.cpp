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
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TestCubeComputeGetL0c2gmConfig : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
constexpr uint64_t L0C2GM_RELU_SHIFT = 0;
constexpr uint64_t L0C2GM_QUANT_SHIFT = 8;
constexpr uint64_t L0C2GM_UNITFLAG_SHIFT = 63;
uint64_t g_fpcConfigValue = 0;

int64_t get_l0c2gm_config_Stub() { return static_cast<int64_t>(g_fpcConfigValue); }

uint64_t build_l0c2gm_config(uint64_t reluUnits = 0, uint64_t quantUnits = 0, uint64_t unitFlag = 0)
{
    return (reluUnits << L0C2GM_RELU_SHIFT) | (quantUnits << L0C2GM_QUANT_SHIFT) | (unitFlag << L0C2GM_UNITFLAG_SHIFT);
}
} // namespace

TEST_F(TestCubeComputeGetL0c2gmConfig, get_l0c2gm_relu_Succ)
{
    constexpr uint64_t expectedValue = 1;
    g_fpcConfigValue = build_l0c2gm_config(expectedValue);

    MOCKER(get_fpc, int64_t()).times(1).will(invoke(get_l0c2gm_config_Stub));

    uint64_t result = asc_get_l0c2gm_relu();
    EXPECT_EQ(expectedValue, result);
    GlobalMockObject::verify();
}

TEST_F(TestCubeComputeGetL0c2gmConfig, get_l0c2gm_prequant_Succ)
{
    constexpr uint64_t expectedValue = 1;
    g_fpcConfigValue = build_l0c2gm_config(0, expectedValue);

    MOCKER(get_fpc, int64_t()).times(1).will(invoke(get_l0c2gm_config_Stub));

    uint64_t result = asc_get_l0c2gm_prequant();
    EXPECT_EQ(expectedValue, result);
    GlobalMockObject::verify();
}

TEST_F(TestCubeComputeGetL0c2gmConfig, get_l0c2gm_unitflag_Succ)
{
    constexpr uint64_t expectedValue = 1;
    g_fpcConfigValue = build_l0c2gm_config(0, 0, expectedValue);

    MOCKER(get_fpc, int64_t()).times(1).will(invoke(get_l0c2gm_config_Stub));

    uint64_t result = asc_get_l0c2gm_unitflag();
    EXPECT_EQ(expectedValue, result);
    GlobalMockObject::verify();
}
