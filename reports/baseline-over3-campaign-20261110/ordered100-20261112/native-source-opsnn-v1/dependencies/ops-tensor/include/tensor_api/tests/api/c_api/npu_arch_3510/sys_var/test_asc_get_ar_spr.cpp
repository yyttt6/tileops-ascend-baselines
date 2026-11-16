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

class TestSysVarGetArSpr : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
int64_t get_ar_Stub() { return 1; }
} // namespace

TEST_F(TestSysVarGetArSpr, c_api_get_ar_spr_Succ)
{
    MOCKER_CPP(get_ar, int64_t()).times(1).will(invoke(get_ar_Stub));
    int64_t ret = asc_get_ar_spr();
    EXPECT_EQ(1, ret);
}
