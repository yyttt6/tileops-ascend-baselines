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

class TestCubeDataMoveCopyL12L0bSparse : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void load_cbuf_to_cb_sp_Stub(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t start_index, uint8_t repeat)
{
    EXPECT_EQ(dst, reinterpret_cast<__cb__ int8_t*>(11));
    EXPECT_EQ((uint64_t)src, 0x0000000100000002);
    EXPECT_EQ(start_index, static_cast<uint16_t>(4));
    EXPECT_EQ(repeat, static_cast<uint8_t>(5));
}
} // namespace

TEST_F(TestCubeDataMoveCopyL12L0bSparse, copy_l12l0b_sparse_Succ)
{
    __cb__ int8_t* dst = reinterpret_cast<__cb__ int8_t*>(11);
    __cbuf__ int8_t* src = reinterpret_cast<__cbuf__ int8_t*>(0x00000002);
    __cbuf__ int8_t* index = reinterpret_cast<__cbuf__ int8_t*>(0x00000001);
    uint16_t start_index = 4;
    uint8_t repeat = 5;
    MOCKER(load_cbuf_to_cb_sp, void(__cb__ int8_t * dst, __cbuf__ int8_t * src, uint16_t start_index, uint8_t repeat))
        .times(1)
        .will(invoke(load_cbuf_to_cb_sp_Stub));

    asc_copy_l12l0b_sparse(dst, src, index, start_index, repeat);
    GlobalMockObject::verify();
}

TEST_F(TestCubeDataMoveCopyL12L0bSparse, copy_l12l0b_sparse_sync_Succ)
{
    __cb__ int8_t* dst = reinterpret_cast<__cb__ int8_t*>(11);
    __cbuf__ int8_t* src = reinterpret_cast<__cbuf__ int8_t*>(0x00000002);
    __cbuf__ int8_t* index = reinterpret_cast<__cbuf__ int8_t*>(0x00000001);
    uint16_t start_index = 4;
    uint8_t repeat = 5;
    MOCKER(load_cbuf_to_cb_sp, void(__cb__ int8_t * dst, __cbuf__ int8_t * src, uint16_t start_index, uint8_t repeat))
        .times(1)
        .will(invoke(load_cbuf_to_cb_sp_Stub));

    asc_copy_l12l0b_sparse_sync(dst, src, index, start_index, repeat);
    GlobalMockObject::verify();
}
