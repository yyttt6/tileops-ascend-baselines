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

class TestCubeDmamoveCopyL12Bt : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void copy_cbuf_to_bt_Stub(
    uint64_t dst_in, __cbuf__ void* src_in, uint16_t conv_control_in, uint16_t n_burst_in, uint16_t len_burst_in,
    uint16_t source_gap_in, uint16_t dst_gap_in)
{
    uint64_t dst = 1;
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(2);
    uint16_t conv_control = 3;
    uint16_t n_burst = 4;
    uint16_t len_burst = 5;
    uint16_t source_gap = 6;
    uint16_t dst_gap = 7;
    EXPECT_EQ(dst, dst_in);
    EXPECT_EQ(src, src_in);
    EXPECT_EQ(conv_control, conv_control_in);
    EXPECT_EQ(n_burst, n_burst_in);
    EXPECT_EQ(len_burst, len_burst_in);
    EXPECT_EQ(source_gap, source_gap_in);
    EXPECT_EQ(dst_gap, dst_gap_in);
}

} // namespace

TEST_F(TestCubeDmamoveCopyL12Bt, asc_copy_l12bt_Succ)
{
    uint64_t dst = 1;
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(2);
    uint16_t conv_control = 3;
    uint16_t n_burst = 4;
    uint16_t len_burst = 5;
    uint16_t source_gap = 6;
    uint16_t dst_gap = 7;

    MOCKER(copy_cbuf_to_bt, void(uint64_t, __cbuf__ void*, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t))
        .times(1)
        .will(invoke(copy_cbuf_to_bt_Stub));

    asc_copy_l12bt(dst, src, conv_control, n_burst, len_burst, source_gap, dst_gap);
    GlobalMockObject::verify();
}
