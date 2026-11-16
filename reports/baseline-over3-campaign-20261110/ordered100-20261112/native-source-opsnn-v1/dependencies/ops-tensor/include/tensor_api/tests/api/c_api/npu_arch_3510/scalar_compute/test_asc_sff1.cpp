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

class TestSysVarSETCTR : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
int64_t get_get_sff1_Stub(uint64_t config) { return 0; }
} // namespace

TEST_F(TestSysVarSETCTR, c_api_get_sff1_Succ)
{
    int64_t input = 1;
    MOCKER_CPP(sff1, int64_t(uint64_t)).times(1).will(invoke(get_get_sff1_Stub));
    asc_ffs(input);
    GlobalMockObject::verify();
}
