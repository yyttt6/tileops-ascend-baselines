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

template <typename DTYPE>
__aicore__ inline void load_gm_to_cb_stub(
    __cb__ DTYPE* dst, __gm__ DTYPE* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap,
    uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    EXPECT_EQ(dst, reinterpret_cast<__cb__ DTYPE*>(11));
    EXPECT_EQ(src, reinterpret_cast<__gm__ DTYPE*>(22));
    EXPECT_EQ(base_idx, static_cast<uint16_t>(33));
    EXPECT_EQ(repeat, static_cast<uint8_t>(44));
    EXPECT_EQ(src_stride, static_cast<uint16_t>(55));
    EXPECT_EQ(dst_gap, static_cast<uint16_t>(66));
    EXPECT_EQ(sid, static_cast<uint16_t>(0));
    EXPECT_EQ(addr_cal_mode, static_cast<addr_cal_mode_t>(addr_cal_mode_t::inc));
}

__aicore__ inline void load_gm_to_cb_s4_stub(
    __cb__ void* dst, __gm__ void* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap,
    uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    EXPECT_EQ(dst, reinterpret_cast<__cb__ void*>(11));
    EXPECT_EQ(src, reinterpret_cast<__gm__ void*>(22));
    EXPECT_EQ(base_idx, static_cast<uint16_t>(33));
    EXPECT_EQ(repeat, static_cast<uint8_t>(44));
    EXPECT_EQ(src_stride, static_cast<uint16_t>(55));
    EXPECT_EQ(dst_gap, static_cast<uint16_t>(66));
    EXPECT_EQ(sid, static_cast<uint16_t>(0));
    EXPECT_EQ(addr_cal_mode, static_cast<addr_cal_mode_t>(addr_cal_mode_t::inc));
}

class TEST_COPY_GM_TO_L0B : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

#define TEST_CUBE_DATAMOVE_COPY_GM2L0B(dtype)                                                                    \
                                                                                                                 \
    TEST_F(TEST_COPY_GM_TO_L0B, TEST_COPY_GM_TO_L0B_##dtype)                                                     \
    {                                                                                                            \
        MOCKER_CPP(                                                                                              \
            load_gm_to_cb,                                                                                       \
            void(__cb__ dtype*, __gm__ dtype*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t)) \
            .times(1)                                                                                            \
            .will(invoke(load_gm_to_cb_stub<dtype>));                                                            \
                                                                                                                 \
        __cb__ dtype* dst = reinterpret_cast<__cb__ dtype*>(11);                                                 \
        __gm__ dtype* src = reinterpret_cast<__gm__ dtype*>(22);                                                 \
                                                                                                                 \
        uint16_t base_idx = static_cast<uint16_t>(33);                                                           \
        uint8_t repeat = static_cast<uint8_t>(44);                                                               \
        uint16_t src_stride = static_cast<uint16_t>(55);                                                         \
        uint8_t dst_gap = static_cast<uint8_t>(66);                                                              \
                                                                                                                 \
        asc_copy_gm2l0b(dst, src, base_idx, repeat, src_stride, dst_gap);                                        \
        GlobalMockObject::verify();                                                                              \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TEST_COPY_GM_TO_L0B, TEST_COPY_GM_TO_L0B_SYNC_##dtype)                                                \
    {                                                                                                            \
        MOCKER_CPP(                                                                                              \
            load_gm_to_cb,                                                                                       \
            void(__cb__ dtype*, __gm__ dtype*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t)) \
            .times(1)                                                                                            \
            .will(invoke(load_gm_to_cb_stub<dtype>));                                                            \
                                                                                                                 \
        __cb__ dtype* dst = reinterpret_cast<__cb__ dtype*>(11);                                                 \
        __gm__ dtype* src = reinterpret_cast<__gm__ dtype*>(22);                                                 \
                                                                                                                 \
        uint16_t base_idx = static_cast<uint16_t>(33);                                                           \
        uint8_t repeat = static_cast<uint8_t>(44);                                                               \
        uint16_t src_stride = static_cast<uint16_t>(55);                                                         \
        uint8_t dst_gap = static_cast<uint8_t>(66);                                                              \
                                                                                                                 \
        asc_copy_gm2l0b_sync(dst, src, base_idx, repeat, src_stride, dst_gap);                                   \
        GlobalMockObject::verify();                                                                              \
    }

// ==========asc_copy_gm2l0b==========
TEST_CUBE_DATAMOVE_COPY_GM2L0B(bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(half);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(float);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(int32_t);
// TEST_CUBE_DATAMOVE_COPY_GM2L0B(int64_t);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(int8_t);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(uint32_t);
TEST_CUBE_DATAMOVE_COPY_GM2L0B(uint8_t);

TEST_F(TEST_COPY_GM_TO_L0B, TEST_COPY_GM_TO_L0B_INT4B_T)
{
    MOCKER_CPP(
        load_gm_to_cb_s4,
        void(__cb__ void*, __gm__ void*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t))
        .times(1)
        .will(invoke(load_gm_to_cb_s4_stub));

    __cb__ int4b_t* dst = reinterpret_cast<__cb__ int4b_t*>(11);
    __gm__ int4b_t* src = reinterpret_cast<__gm__ int4b_t*>(22);

    uint16_t base_idx = static_cast<uint16_t>(33);
    uint8_t repeat = static_cast<uint8_t>(44);
    uint16_t src_stride = static_cast<uint16_t>(55);
    uint8_t dst_gap = static_cast<uint8_t>(66);

    asc_copy_gm2l0b(dst, src, base_idx, repeat, src_stride, dst_gap);
    GlobalMockObject::verify();
}

TEST_F(TEST_COPY_GM_TO_L0B, TEST_COPY_GM_TO_L0B_SYNC_INT4B_T)
{
    MOCKER_CPP(
        load_gm_to_cb_s4,
        void(__cb__ void*, __gm__ void*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t))
        .times(1)
        .will(invoke(load_gm_to_cb_s4_stub));

    __cb__ int4b_t* dst = reinterpret_cast<__cb__ int4b_t*>(11);
    __gm__ int4b_t* src = reinterpret_cast<__gm__ int4b_t*>(22);

    uint16_t base_idx = static_cast<uint16_t>(33);
    uint8_t repeat = static_cast<uint8_t>(44);
    uint16_t src_stride = static_cast<uint16_t>(55);
    uint8_t dst_gap = static_cast<uint8_t>(66);

    asc_copy_gm2l0b_sync(dst, src, base_idx, repeat, src_stride, dst_gap);
    GlobalMockObject::verify();
}
