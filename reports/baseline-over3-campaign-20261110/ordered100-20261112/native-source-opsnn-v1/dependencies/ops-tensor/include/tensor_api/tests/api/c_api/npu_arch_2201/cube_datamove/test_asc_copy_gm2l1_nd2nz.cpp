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

#define TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(class_name, c_api_name, cce_name, datatype)                                 \
                                                                                                                            \
    class TestCubeDmamove##class_name##datatype : public testing::Test {                                                    \
    protected:                                                                                                              \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                                       \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                                    \
    };                                                                                                                      \
                                                                                                                            \
    namespace {                                                                                                             \
                                                                                                                            \
    void cce_name##_##datatype##datatype##_uint8_t_uint16_uint16_uint16_uint16_uint16_uint16_uint16_uint16_t_Stub(          \
        __cbuf__ datatype* dst, __gm__ datatype* src, uint8_t sid, uint16_t nd_num, uint16_t n_value,                       \
        uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_co_stride,                   \
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)                                                            \
    {                                                                                                                       \
        EXPECT_EQ(dst, reinterpret_cast<__cbuf__ datatype*>(11));                                                           \
        EXPECT_EQ(src, reinterpret_cast<__gm__ datatype*>(22));                                                             \
        EXPECT_EQ(sid, static_cast<uint8_t>(0));                                                                            \
        EXPECT_EQ(nd_num, static_cast<uint16_t>(1));                                                                        \
        EXPECT_EQ(n_value, static_cast<uint16_t>(1));                                                                       \
        EXPECT_EQ(d_value, static_cast<uint16_t>(1));                                                                       \
        EXPECT_EQ(src_nd_matrix_stride, static_cast<uint16_t>(0));                                                          \
        EXPECT_EQ(src_d_value, static_cast<uint16_t>(1));                                                                   \
        EXPECT_EQ(dst_nz_co_stride, static_cast<uint16_t>(1));                                                              \
        EXPECT_EQ(dst_nz_n_stride, static_cast<uint16_t>(1));                                                               \
        EXPECT_EQ(dst_nz_matrix_stride, static_cast<uint16_t>(1));                                                          \
    }                                                                                                                       \
    }                                                                                                                       \
                                                                                                                            \
    TEST_F(TestCubeDmamove##class_name##datatype, c_api_name##_CopyConfig_Succ)                                             \
    {                                                                                                                       \
        __cbuf__ datatype* dst = reinterpret_cast<__cbuf__ datatype*>(11);                                                  \
        __gm__ datatype* src = reinterpret_cast<__gm__ datatype*>(22);                                                      \
                                                                                                                            \
        uint16_t nd_num = static_cast<uint16_t>(1);                                                                         \
        uint16_t n_value = static_cast<uint16_t>(1);                                                                        \
        uint16_t d_value = static_cast<uint16_t>(1);                                                                        \
        uint16_t src_nd_matrix_stride = static_cast<uint16_t>(0);                                                           \
        uint16_t src_d_value = static_cast<uint16_t>(1);                                                                    \
        uint16_t dst_nz_co_stride = static_cast<uint16_t>(1);                                                               \
        uint16_t dst_nz_n_stride = static_cast<uint16_t>(1);                                                                \
        uint16_t dst_nz_matrix_stride = static_cast<uint16_t>(1);                                                           \
                                                                                                                            \
        MOCKER_CPP(                                                                                                         \
            cce_name, void(                                                                                                 \
                          __cbuf__ datatype*, __gm__ datatype*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t,            \
                          uint16_t, uint16_t, uint16_t, uint16_t))                                                          \
            .times(1)                                                                                                       \
            .will(invoke(                                                                                                   \
                cce_name##_##datatype##datatype##_uint8_t_uint16_uint16_uint16_uint16_uint16_uint16_uint16_uint16_t_Stub)); \
                                                                                                                            \
        c_api_name(                                                                                                         \
            dst, src, nd_num, n_value, d_value, src_nd_matrix_stride, src_d_value, dst_nz_co_stride, dst_nz_n_stride,       \
            dst_nz_matrix_stride);                                                                                          \
        GlobalMockObject::verify();                                                                                         \
    }                                                                                                                       \
                                                                                                                            \
    TEST_F(TestCubeDmamove##class_name##datatype, c_api_name##_sync_Succ)                                                   \
    {                                                                                                                       \
        __cbuf__ datatype* dst = reinterpret_cast<__cbuf__ datatype*>(11);                                                  \
        __gm__ datatype* src = reinterpret_cast<__gm__ datatype*>(22);                                                      \
                                                                                                                            \
        uint16_t nd_num = static_cast<uint16_t>(1);                                                                         \
        uint16_t n_value = static_cast<uint16_t>(1);                                                                        \
        uint16_t d_value = static_cast<uint16_t>(1);                                                                        \
        uint16_t src_nd_matrix_stride = static_cast<uint16_t>(0);                                                           \
        uint16_t src_d_value = static_cast<uint16_t>(1);                                                                    \
        uint16_t dst_nz_co_stride = static_cast<uint16_t>(1);                                                               \
        uint16_t dst_nz_n_stride = static_cast<uint16_t>(1);                                                                \
        uint16_t dst_nz_matrix_stride = static_cast<uint16_t>(1);                                                           \
                                                                                                                            \
        MOCKER_CPP(                                                                                                         \
            cce_name, void(                                                                                                 \
                          __cbuf__ datatype*, __gm__ datatype*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t,            \
                          uint16_t, uint16_t, uint16_t, uint16_t))                                                          \
            .times(1)                                                                                                       \
            .will(invoke(                                                                                                   \
                cce_name##_##datatype##datatype##_uint8_t_uint16_uint16_uint16_uint16_uint16_uint16_uint16_uint16_t_Stub)); \
                                                                                                                            \
        c_api_name(                                                                                                         \
            dst, src, nd_num, n_value, d_value, src_nd_matrix_stride, src_d_value, dst_nz_co_stride, dst_nz_n_stride,       \
            dst_nz_matrix_stride);                                                                                          \
        GlobalMockObject::verify();                                                                                         \
    }

// ==========asc_copy_gm2l1_nd2nz b8==========
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(CopyGM2L1ND2NZB8, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b8, int8_t);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB8, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b8, uint8_t);
// ==========asc_copy_gm2l1_nd2nz b16==========
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB16, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b16, bfloat16_t);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(CopyGM2L1ND2NZB16, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b16, half);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB16, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b16, int16_t);
// ==========asc_copy_gm2l1_nd2nz b32s==========
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB32s, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b32s, float);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB32s, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b32s, int32_t);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF_ND2NZ(
    CopyGM2L1ND2NZB32s, asc_copy_gm2l1_nd2nz, copy_gm_to_cbuf_multi_nd2nz_b32s, uint32_t);
