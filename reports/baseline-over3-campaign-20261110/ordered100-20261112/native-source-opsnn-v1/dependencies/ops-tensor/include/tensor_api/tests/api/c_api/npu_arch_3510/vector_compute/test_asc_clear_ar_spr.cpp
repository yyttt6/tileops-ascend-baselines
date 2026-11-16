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

class TestVectorComputeClearArSpr : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void sprclr_Stub(Literal spr_id) { EXPECT_EQ(spr_id, SPR_AR); }
} // namespace

TEST_F(TestVectorComputeClearArSpr, c_api_clear_ar_spr_Succ)
{
    MOCKER_CPP(sprclr, void(Literal)).times(1).will(invoke(sprclr_Stub));
    asc_clear_ar_spr();
    GlobalMockObject::verify();
}
