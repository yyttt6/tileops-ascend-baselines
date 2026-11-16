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

class TestSetVaRegCAPI : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void asc_set_va_reg_stub(ub_addr8_t addr, uint64_t* vaRegArray)
{
    EXPECT_EQ(addr, ub_addr8_t::VA0);
    EXPECT_EQ(vaRegArray[0], (uint64_t)11);
}
} // namespace

TEST_F(TestSetVaRegCAPI, c_api_set_va_reg_Succ)
{
    ub_addr8_t addr = ub_addr8_t::VA0;
    __ubuf__ half* src[16] = {reinterpret_cast<__ubuf__ half*>(11)};

    MOCKER_CPP(set_va_reg_sb, void(ub_addr8_t, uint64_t*)).times(1).will(invoke(asc_set_va_reg_stub));

    asc_set_va_reg(addr, src);
    GlobalMockObject::verify();
}
