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

class TestScalarComputeFfs : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
int64_t sff1_Stub(uint64_t in)
{
    uint64_t value = 10;
    int64_t pos = 2;
    EXPECT_EQ(value, in);
    return pos;
}
} // namespace

TEST_F(TestScalarComputeFfs, c_api_ffs_Succ)
{
    MOCKER(sff1).times(1).will(invoke(sff1_Stub));
    uint64_t in = 10;
    int64_t pos = 2;

    int64_t ret = asc_ffs(in);
    EXPECT_EQ(pos, ret);
    GlobalMockObject::verify();
}
