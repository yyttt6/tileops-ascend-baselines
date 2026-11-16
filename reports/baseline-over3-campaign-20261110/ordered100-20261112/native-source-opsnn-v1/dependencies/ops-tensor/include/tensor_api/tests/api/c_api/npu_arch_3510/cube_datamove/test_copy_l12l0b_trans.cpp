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
#include "c_api/asc_simd.h"

#define TEST_CUBE_DATAMOVE_L12L0_TRANS(class_name, c_api_name, cce_name, data_type)                                  \
                                                                                                                     \
    class TestCubeDatamove##c_api_name##cce_name##data_type##CApi : public testing::Test {                           \
    protected:                                                                                                       \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                                \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                             \
    };                                                                                                               \
                                                                                                                     \
    void mock##c_api_name##cce_name##_##data_type(                                                                   \
        __cb__ data_type* dst, __cbuf__ data_type* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride,      \
        uint16_t dst_gap, bool addrmode, uint16_t dst_frac_gap, uint16_t src_frac_gap)                               \
    {                                                                                                                \
        EXPECT_EQ(dst, reinterpret_cast<data_type*>(11));                                                            \
        EXPECT_EQ(src, reinterpret_cast<data_type*>(22));                                                            \
        EXPECT_EQ(index_id, static_cast<uint16_t>(33));                                                              \
        EXPECT_EQ(repeat, static_cast<uint8_t>(44));                                                                 \
        EXPECT_EQ(src_stride, static_cast<uint16_t>(55));                                                            \
        EXPECT_EQ(dst_gap, static_cast<uint16_t>(66));                                                               \
        EXPECT_EQ(dst_frac_gap, static_cast<uint16_t>(77));                                                          \
        EXPECT_EQ(src_frac_gap, static_cast<uint16_t>(88));                                                          \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestCubeDatamove##c_api_name##cce_name##data_type##CApi, c_api_name##_Succ)                               \
    {                                                                                                                \
        __cb__ data_type* dst = reinterpret_cast<__cb__ data_type*>(11);                                             \
        __cbuf__ data_type* src = reinterpret_cast<__cbuf__ data_type*>(22);                                         \
                                                                                                                     \
        uint16_t index_id = static_cast<uint16_t>(33);                                                               \
        uint8_t repeat = static_cast<uint8_t>(44);                                                                   \
        uint16_t src_stride = static_cast<uint16_t>(55);                                                             \
        uint16_t dst_gap = static_cast<uint16_t>(66);                                                                \
        uint16_t dst_frac_gap = static_cast<uint16_t>(77);                                                           \
        uint16_t src_frac_gap = static_cast<uint16_t>(88);                                                           \
                                                                                                                     \
        MOCKER_CPP(                                                                                                  \
            cce_name, void(data_type*, data_type*, uint16_t, uint8_t, uint16_t, uint16_t, bool, uint16_t, uint16_t)) \
            .times(1)                                                                                                \
            .will(invoke(mock##c_api_name##cce_name##_##data_type));                                                 \
        c_api_name(dst, src, index_id, repeat, src_stride, dst_gap, dst_frac_gap, src_frac_gap);                     \
        GlobalMockObject::verify();                                                                                  \
    }

TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, bfloat16_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, float8_e4m3_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, float8_e5m2_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, half);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, float);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, hifloat8_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, int32_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, int8_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, uint32_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, uint8_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose_s4, fp4x2_e2m1_t);
TEST_CUBE_DATAMOVE_L12L0_TRANS(CopyL12L0Trans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose_s4, fp4x2_e1m2_t);
