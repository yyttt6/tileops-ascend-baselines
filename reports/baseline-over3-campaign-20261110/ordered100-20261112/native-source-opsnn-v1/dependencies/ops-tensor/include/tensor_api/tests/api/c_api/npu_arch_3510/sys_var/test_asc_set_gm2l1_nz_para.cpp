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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

class TestSysVarSetGm2l1NzPara : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void set_set_gm2l1_nz_para_Stub(uint64_t config) { EXPECT_EQ(config, static_cast<uint64_t>(1)); }
} // namespace

TEST_F(TestSysVarSetGm2l1NzPara, c_api_set_gm2l1_nz_para_Succ)
{
    int64_t input = 1;
    MOCKER_CPP(set_mte2_nz_para, void(uint64_t)).times(1).will(invoke(set_set_gm2l1_nz_para_Stub));
    asc_set_gm2l1_nz_para(input);
    asc_init();
    GlobalMockObject::verify();
}
