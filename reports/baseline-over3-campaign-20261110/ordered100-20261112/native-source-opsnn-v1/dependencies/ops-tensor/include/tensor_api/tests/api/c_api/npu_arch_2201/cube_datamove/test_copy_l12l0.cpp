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

#define TEST_CUBE_DATAMOVE_COPY_L12L0_2D(class_name, c_api_name, cce_name, dst_pos, data_type, cce_type)               \
                                                                                                                       \
    class TestCubeDmamove##class_name##data_type : public testing::Test {                                              \
    protected:                                                                                                         \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                                  \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                               \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
                                                                                                                       \
    void cce_name##_##data_type##_Stub(                                                                                \
        dst_pos cce_type* dst, __cbuf__ cce_type* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride,      \
        uint16_t dst_gap, uint8_t sid, bool transpose, addr_cal_mode_t addr_mode)                                      \
    {                                                                                                                  \
        EXPECT_EQ(dst, reinterpret_cast<dst_pos cce_type*>(11));                                                       \
        EXPECT_EQ(src, reinterpret_cast<__cbuf__ cce_type*>(22));                                                      \
        EXPECT_EQ(start_index, static_cast<uint16_t>(0));                                                              \
        EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                                    \
        EXPECT_EQ(src_stride, static_cast<uint16_t>(8));                                                               \
        EXPECT_EQ(dst_gap, static_cast<uint16_t>(0));                                                                  \
        EXPECT_EQ(sid, static_cast<uint8_t>(0));                                                                       \
        EXPECT_EQ(transpose, false);                                                                                   \
        EXPECT_EQ(addr_mode, addr_cal_mode_t::inc);                                                                    \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestCubeDmamove##class_name##data_type, c_api_name##_LoadImageConfig_Succ)                                  \
    {                                                                                                                  \
        dst_pos data_type* dst = reinterpret_cast<dst_pos data_type*>(11);                                             \
        __cbuf__ data_type* src = reinterpret_cast<__cbuf__ data_type*>(22);                                           \
                                                                                                                       \
        uint16_t start_index = static_cast<uint16_t>(0);                                                               \
        uint8_t repeat = static_cast<uint8_t>(1);                                                                      \
        uint16_t src_stride = static_cast<uint16_t>(8);                                                                \
        uint16_t dst_gap = static_cast<uint16_t>(0);                                                                   \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(                                                                                            \
                          dst_pos cce_type*, __cbuf__ cce_type*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, bool, \
                          addr_cal_mode_t))                                                                            \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src, start_index, repeat, src_stride, dst_gap);                                                \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestCubeDmamove##class_name##data_type, c_api_name##_sync_Succ)                                             \
    {                                                                                                                  \
        dst_pos data_type* dst = reinterpret_cast<dst_pos data_type*>(11);                                             \
        __cbuf__ data_type* src = reinterpret_cast<__cbuf__ data_type*>(22);                                           \
                                                                                                                       \
        uint16_t start_index = static_cast<uint16_t>(0);                                                               \
        uint8_t repeat = static_cast<uint8_t>(1);                                                                      \
        uint16_t src_stride = static_cast<uint16_t>(8);                                                                \
        uint16_t dst_gap = static_cast<uint16_t>(0);                                                                   \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(                                                                                            \
                          dst_pos cce_type*, __cbuf__ cce_type*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, bool, \
                          addr_cal_mode_t))                                                                            \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name##_sync(dst, src, start_index, repeat, src_stride, dst_gap);                                         \
        GlobalMockObject::verify();                                                                                    \
    }

// ==========asc_copy_l12l0a==========
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca_s4, __ca__, int4b_t, void);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, int8_t, int8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, uint8_t, uint8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, half, half);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, bfloat16_t, bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, int32_t, int32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, uint32_t, uint32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0A2DCApi, asc_copy_l12l0a, load_cbuf_to_ca, __ca__, float, float);
// ==========asc_copy_l12l0b==========
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb_s4, __cb__, int4b_t, void);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, int8_t, int8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, uint8_t, uint8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, half, half);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, bfloat16_t, bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, int32_t, int32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, uint32_t, uint32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0_2D(CopyL12L0B2DCApi, asc_copy_l12l0b, load_cbuf_to_cb, __cb__, float, float);
