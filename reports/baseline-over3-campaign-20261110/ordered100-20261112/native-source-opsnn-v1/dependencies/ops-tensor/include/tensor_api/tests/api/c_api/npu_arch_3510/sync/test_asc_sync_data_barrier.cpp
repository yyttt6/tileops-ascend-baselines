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
#include "include/c_api/asc_simd.h"

class TestAscSyncDataBarrierCAPI : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void dsb_stub(mem_dsb_t arg) { EXPECT_EQ(mem_dsb_t::DSB_ALL, arg); }
} // namespace

TEST_F(TestAscSyncDataBarrierCAPI, c_api_asc_sync_data_barrier_succ)
{
    mem_dsb_t arg = mem_dsb_t::DSB_ALL;
    MOCKER_CPP(dsb, void(mem_dsb_t)).times(1).will(invoke(dsb_stub));

    asc_sync_data_barrier(arg);
    GlobalMockObject::verify();
}
