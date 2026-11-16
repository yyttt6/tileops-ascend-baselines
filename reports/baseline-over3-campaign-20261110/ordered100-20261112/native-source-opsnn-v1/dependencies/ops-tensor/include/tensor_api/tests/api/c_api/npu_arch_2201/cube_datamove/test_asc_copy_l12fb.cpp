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

class TestCubeDmamoveCopyL12FbCAPI : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void copy_cbuf_to_fbuf_Stub(
    __fbuf__ void* dst_in, __cbuf__ void* src_in, uint16_t n_burst_in, uint16_t len_burst_in, uint16_t src_gap_size_in,
    uint16_t dst_gap_size_in)
{
    __fbuf__ void* dst = reinterpret_cast<__fbuf__ void*>(1);
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(2);
    uint16_t n_burst = 3;
    uint16_t len_burst = 4;
    uint16_t src_gap_size = 5;
    uint16_t dst_gap_size = 6;
    EXPECT_EQ(dst, dst_in);
    EXPECT_EQ(src, src_in);
    EXPECT_EQ(n_burst, n_burst_in);
    EXPECT_EQ(len_burst, len_burst_in);
    EXPECT_EQ(src_gap_size, src_gap_size_in);
    EXPECT_EQ(dst_gap_size, dst_gap_size_in);
}

} // namespace

TEST_F(TestCubeDmamoveCopyL12FbCAPI, c_api_asc_copy_l12fb_Succ)
{
    __fbuf__ void* dst = reinterpret_cast<__fbuf__ void*>(1);
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(2);
    uint16_t n_burst = 3;
    uint16_t len_burst = 4;
    uint16_t src_gap_size = 5;
    uint16_t dst_gap_size = 6;

    MOCKER(copy_cbuf_to_fbuf, void(__fbuf__ void*, __cbuf__ void*, uint16_t, uint16_t, uint16_t, uint16_t))
        .times(1)
        .will(invoke(copy_cbuf_to_fbuf_Stub));

    asc_copy_l12fb(dst, src, n_burst, len_burst, src_gap_size, dst_gap_size);
    GlobalMockObject::verify();
}
