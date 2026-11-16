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

class TestMmadCAPI : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

namespace {
void mad_Stub(
    __cc__ float* c, __ca__ float* a, __cb__ float* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,
    bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    EXPECT_EQ(c, reinterpret_cast<__cc__ float*>(1));
    EXPECT_EQ(a, reinterpret_cast<__ca__ float*>(2));
    EXPECT_EQ(b, reinterpret_cast<__cb__ float*>(3));
    EXPECT_EQ(m, static_cast<uint16_t>(4));
    EXPECT_EQ(k, static_cast<uint16_t>(5));
    EXPECT_EQ(n, static_cast<uint16_t>(6));
    EXPECT_EQ(unitFlag, static_cast<uint8_t>(1));
    EXPECT_EQ(cmatrixSource, false);
    EXPECT_EQ(cmatrixInitVal, true);
}

void mad_sync_Stub(
    __cc__ float* c, __ca__ float* a, __cb__ float* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,
    bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    EXPECT_EQ(c, reinterpret_cast<__cc__ float*>(1));
    EXPECT_EQ(a, reinterpret_cast<__ca__ float*>(2));
    EXPECT_EQ(b, reinterpret_cast<__cb__ float*>(3));
    EXPECT_EQ(m, static_cast<uint16_t>(4));
    EXPECT_EQ(k, static_cast<uint16_t>(5));
    EXPECT_EQ(n, static_cast<uint16_t>(6));
    EXPECT_EQ(unitFlag, static_cast<uint8_t>(1));
    EXPECT_EQ(cmatrixSource, false);
    EXPECT_EQ(cmatrixInitVal, true);
}

} // namespace

TEST_F(TestMmadCAPI, mmad_Succ)
{
    __cc__ float* c = reinterpret_cast<__cc__ float*>(1);
    __ca__ float* a = reinterpret_cast<__ca__ float*>(2);
    __cb__ float* b = reinterpret_cast<__cb__ float*>(3);
    uint16_t m = 4;
    uint16_t k = 5;
    uint16_t n = 6;
    uint8_t unitFlag = 1;
    bool kDirectionAlign = true;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
    MOCKER(
        mad, void(
                 __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height,
                 uint16_t right_width, uint16_t n_dim, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source,
                 bool c_matrix_init_val))
        .times(1)
        .will(invoke(mad_Stub));

    asc_mmad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
    GlobalMockObject::verify();
}

TEST_F(TestMmadCAPI, mmad_sync_Succ)
{
    __cc__ float* c = reinterpret_cast<__cc__ float*>(1);
    __ca__ float* a = reinterpret_cast<__ca__ float*>(2);
    __cb__ float* b = reinterpret_cast<__cb__ float*>(3);
    uint16_t m = 4;
    uint16_t k = 5;
    uint16_t n = 6;
    uint8_t unitFlag = 1;
    bool kDirectionAlign = true;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
    MOCKER(
        mad, void(
                 __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height,
                 uint16_t right_width, uint16_t n_dim, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source,
                 bool c_matrix_init_val))
        .times(1)
        .will(invoke(mad_sync_Stub));

    asc_mmad_sync(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
    GlobalMockObject::verify();
}
