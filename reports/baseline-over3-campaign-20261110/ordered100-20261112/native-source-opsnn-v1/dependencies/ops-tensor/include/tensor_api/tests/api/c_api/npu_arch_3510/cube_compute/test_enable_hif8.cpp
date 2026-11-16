/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class TestEnableHif8CAPI : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {

constexpr uint64_t CTRL_VAL = 0x12345678ULL;
constexpr int32_t HIF8_MODE_BIT = 45;
constexpr uint64_t HIF8_MODE_BIT_MASK = 1ULL << HIF8_MODE_BIT;

int64_t get_ctrl_Stub() { return static_cast<int64_t>(CTRL_VAL); }

uint64_t sbitset1_hif8_mode_stub(uint64_t bits, int64_t idx)
{
    EXPECT_EQ(CTRL_VAL, bits);
    EXPECT_EQ(HIF8_MODE_BIT, idx);
    return CTRL_VAL | HIF8_MODE_BIT_MASK;
}

void set_ctrl_hif8_mode_stub(uint64_t ctrl_val) { EXPECT_EQ(CTRL_VAL | HIF8_MODE_BIT_MASK, ctrl_val); }

} // namespace

TEST_F(TestEnableHif8CAPI, c_api_EnableHif8_Success)
{
    MOCKER_CPP(get_ctrl, int64_t()).times(1).will(invoke(get_ctrl_Stub));

    MOCKER_CPP(sbitset1, uint64_t(uint64_t, int64_t)).times(1).will(invoke(sbitset1_hif8_mode_stub));

    MOCKER_CPP(set_ctrl, void(uint64_t ctrl_val)).times(1).will(invoke(set_ctrl_hif8_mode_stub));

    asc_enable_hif8();
    GlobalMockObject::verify();
}
