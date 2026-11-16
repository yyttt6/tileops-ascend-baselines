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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

class TestScalarComputeSflbitsCApi : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
int64_t sflbits_Stub(int64_t value)
{
    EXPECT_EQ(0, value);
    return -1;
}
} // namespace

TEST_F(TestScalarComputeSflbitsCApi, sflbits_uint64_t_Succ)
{
    int64_t res = 0;
    int64_t value = 0;
    MOCKER(sflbits).times(1).will(invoke(sflbits_Stub));

    res = asc_sflbits(value);
    EXPECT_EQ(res, -1);
    GlobalMockObject::verify();
}
