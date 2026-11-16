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

class TestZeroBitsCnt : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
int64_t zero_bits_cnt_Stub(uint64_t config) { return 0; }
} // namespace

TEST_F(TestZeroBitsCnt, c_api_zero_bits_cnt_Succ)
{
    int64_t input = 1;
    MOCKER_CPP(bcnt0, int64_t(uint64_t)).times(1).will(invoke(zero_bits_cnt_Stub));
    asc_zero_bits_cnt(input);
    GlobalMockObject::verify();
}
