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

#define TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(class_name, c_api_name, cce_name, data_type)                                 \
                                                                                                                       \
    class TestCubeDatamove##class_name##data_type##CApi : public testing::Test {                                       \
    protected:                                                                                                         \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                                  \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                               \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
                                                                                                                       \
    void cce_name##_##data_type##_Stub(                                                                                \
        uint64_t dst, __cbuf__ void* src, uint16_t x_start_pos, uint16_t y_start_pos, uint8_t x_step, uint8_t y_step,  \
        uint16_t src_stride, uint16_t dst_stride)                                                                      \
    {                                                                                                                  \
        EXPECT_EQ(dst, static_cast<uint64_t>(0));                                                                      \
        EXPECT_EQ(src, reinterpret_cast<__cbuf__ void*>(22));                                                          \
        EXPECT_EQ(x_start_pos, static_cast<uint16_t>(0));                                                              \
        EXPECT_EQ(y_start_pos, static_cast<uint16_t>(0));                                                              \
        EXPECT_EQ(x_step, static_cast<uint8_t>(0));                                                                    \
        EXPECT_EQ(y_step, static_cast<uint8_t>(0));                                                                    \
        EXPECT_EQ(src_stride, static_cast<uint16_t>(8));                                                               \
        EXPECT_EQ(dst_stride, static_cast<uint16_t>(8));                                                               \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestCubeDatamove##class_name##data_type##CApi, c_api_name##_Succ)                                           \
    {                                                                                                                  \
        __ca__ data_type* dst = reinterpret_cast<__ca__ data_type*>(11);                                               \
        __cbuf__ fp8_e8m0_t* src = reinterpret_cast<__cbuf__ fp8_e8m0_t*>(22);                                         \
                                                                                                                       \
        uint16_t x_start_pos = static_cast<uint16_t>(0);                                                               \
        uint16_t y_start_pos = static_cast<uint16_t>(0);                                                               \
        uint8_t x_step = static_cast<uint8_t>(0);                                                                      \
        uint8_t y_step = static_cast<uint8_t>(0);                                                                      \
        uint16_t src_stride = static_cast<uint16_t>(8);                                                                \
        uint16_t dst_stride = static_cast<uint16_t>(8);                                                                \
                                                                                                                       \
        MOCKER_CPP(cce_name, void(uint64_t, __cbuf__ void*, uint16_t, uint16_t, uint8_t, uint8_t, uint16_t, uint16_t)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        uint64_t mx_dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;                           \
        c_api_name(mx_dst_addr, src, x_start_pos, y_start_pos, x_step, y_step, src_stride, dst_stride);                \
        GlobalMockObject::verify();                                                                                    \
    }

TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0AMX, asc_copy_l12l0a_mx, load_cbuf_to_ca_mx, fp4x2_e2m1_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0AMX, asc_copy_l12l0a_mx, load_cbuf_to_ca_mx, fp4x2_e1m2_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0AMX, asc_copy_l12l0a_mx, load_cbuf_to_ca_mx, fp8_e5m2_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0AMX, asc_copy_l12l0a_mx, load_cbuf_to_ca_mx, fp8_e4m3fn_t);

TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0BMX, asc_copy_l12l0b_mx, load_cbuf_to_cb_mx, fp4x2_e2m1_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0BMX, asc_copy_l12l0b_mx, load_cbuf_to_cb_mx, fp4x2_e1m2_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0BMX, asc_copy_l12l0b_mx, load_cbuf_to_cb_mx, fp8_e5m2_t);
TEST_CUBE_DATAMOVE_COPY_L12L0A_MX(CopyL12L0BMX, asc_copy_l12l0b_mx, load_cbuf_to_cb_mx, fp8_e4m3fn_t);
