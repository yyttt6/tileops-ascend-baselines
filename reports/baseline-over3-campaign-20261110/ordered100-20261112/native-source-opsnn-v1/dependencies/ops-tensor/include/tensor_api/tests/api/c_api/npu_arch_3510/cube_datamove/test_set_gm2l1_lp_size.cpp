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

class TestCubeDatamoveSetGM2L1LoopSize : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void set_loop_size_outtol1_Stub(uint64_t config)
{
    uint64_t conf = (321ul << 21) | 123ul;
    EXPECT_EQ(conf, config);
}
} // namespace

TEST_F(TestCubeDatamoveSetGM2L1LoopSize, set_gm2l1_lp_size_Succ)
{
    MOCKER(set_loop_size_outtol1, void(uint64_t)).times(1).will(invoke(set_loop_size_outtol1_Stub));
    uint64_t loop1_size = 123;
    uint64_t loop2_size = 321;

    asc_set_gm2l1_loop_size(loop1_size, loop2_size);
    GlobalMockObject::verify();
}
