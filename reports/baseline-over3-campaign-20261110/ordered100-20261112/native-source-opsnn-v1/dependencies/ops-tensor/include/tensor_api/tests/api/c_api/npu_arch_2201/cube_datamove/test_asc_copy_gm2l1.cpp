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

#define TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(class_name, c_api_name, cce_name)                                        \
                                                                                                                   \
    class TestCubeDmamove##class_name##c_api_name : public testing::Test {                                         \
    protected:                                                                                                     \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                              \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                           \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
                                                                                                                   \
    void cce_name##_##c_api_name##_uint64_t_Stub(                                                                  \
        __cbuf__ void* dst, __gm__ void* src, uint8_t sid, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, \
        uint16_t dst_gap, pad_t pad_mode)                                                                          \
    {                                                                                                              \
        EXPECT_EQ(dst, reinterpret_cast<__cbuf__ void*>(11));                                                      \
        EXPECT_EQ(src, reinterpret_cast<__gm__ void*>(22));                                                        \
        EXPECT_EQ(sid, static_cast<uint8_t>(0));                                                                   \
        EXPECT_EQ(n_burst, static_cast<uint16_t>(1));                                                              \
        EXPECT_EQ(len_burst, static_cast<uint16_t>(1));                                                            \
        EXPECT_EQ(src_gap, static_cast<uint16_t>(0));                                                              \
        EXPECT_EQ(dst_gap, static_cast<uint16_t>(0));                                                              \
    }                                                                                                              \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestCubeDmamove##class_name##c_api_name, c_api_name##_CopyConfig_Succ)                                  \
    {                                                                                                              \
        __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(11);                                                 \
        __gm__ void* src = reinterpret_cast<__gm__ void*>(22);                                                     \
                                                                                                                   \
        uint16_t n_burst = static_cast<uint16_t>(1);                                                               \
        uint16_t len_burst = static_cast<uint16_t>(1);                                                             \
        uint16_t src_gap = static_cast<uint16_t>(0);                                                               \
        uint16_t dst_gap = static_cast<uint16_t>(0);                                                               \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(__cbuf__ void*, __gm__ void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))  \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##c_api_name##_uint64_t_Stub));                                                \
                                                                                                                   \
        c_api_name(dst, src, n_burst, len_burst, src_gap, dst_gap);                                                \
        GlobalMockObject::verify();                                                                                \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestCubeDmamove##class_name##c_api_name, c_api_name##_size_Succ)                                        \
    {                                                                                                              \
        __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(11);                                                 \
        __gm__ void* src = reinterpret_cast<__gm__ void*>(22);                                                     \
        uint32_t size = static_cast<uint32_t>(44);                                                                 \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(__cbuf__ void*, __gm__ void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))  \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##c_api_name##_uint64_t_Stub));                                                \
                                                                                                                   \
        c_api_name(dst, src, size);                                                                                \
        GlobalMockObject::verify();                                                                                \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestCubeDmamove##class_name##c_api_name, c_api_name##_sync_Succ)                                        \
    {                                                                                                              \
        __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(11);                                                 \
        __gm__ void* src = reinterpret_cast<__gm__ void*>(22);                                                     \
        uint32_t size = static_cast<uint32_t>(44);                                                                 \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(__cbuf__ void*, __gm__ void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))  \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##c_api_name##_uint64_t_Stub));                                                \
        c_api_name##_sync(dst, src, size);                                                                         \
        GlobalMockObject::verify();                                                                                \
    }

// ==========asc_copy_gm2l1==========
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_NONE, asc_copy_gm2l1, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD1, asc_copy_gm2l1_pad1, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD2, asc_copy_gm2l1_pad2, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD3, asc_copy_gm2l1_pad3, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD4, asc_copy_gm2l1_pad4, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD5, asc_copy_gm2l1_pad5, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD6, asc_copy_gm2l1_pad6, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD7, asc_copy_gm2l1_pad7, copy_gm_to_cbuf);
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1_PAD8, asc_copy_gm2l1_pad8, copy_gm_to_cbuf);

template <typename DTYPE>
__aicore__ inline void load_gm_to_cbuf_stub(
    __cbuf__ DTYPE* dst, __gm__ DTYPE* src, uint16_t base_idx, uint8_t repeat, uint16_t src_gap, uint16_t dst_gap,
    uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    EXPECT_EQ(dst, reinterpret_cast<__cbuf__ DTYPE*>(11));
    EXPECT_EQ(src, reinterpret_cast<__gm__ DTYPE*>(22));
    EXPECT_EQ(base_idx, static_cast<uint16_t>(33));
    EXPECT_EQ(repeat, static_cast<uint8_t>(44));
    EXPECT_EQ(src_gap, static_cast<uint16_t>(55));
    EXPECT_EQ(dst_gap, static_cast<uint16_t>(66));
    EXPECT_EQ(sid, static_cast<uint8_t>(0));
    EXPECT_EQ(addr_cal_mode, static_cast<addr_cal_mode_t>(addr_cal_mode_t::inc));
}

class TEST_COPY_GM_TO_L1 : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

#define TEST_CUBE_DATAMOVE_COPY_GM2L1(dtype)                                                                       \
                                                                                                                   \
    TEST_F(TEST_COPY_GM_TO_L1, TEST_COPY_GM_TO_L1_##dtype)                                                         \
    {                                                                                                              \
        MOCKER_CPP(                                                                                                \
            load_gm_to_cbuf,                                                                                       \
            void(__cbuf__ dtype*, __gm__ dtype*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t)) \
            .times(1)                                                                                              \
            .will(invoke(load_gm_to_cbuf_stub<dtype>));                                                            \
                                                                                                                   \
        __cbuf__ dtype* dst = reinterpret_cast<__cbuf__ dtype*>(11);                                               \
        __gm__ dtype* src = reinterpret_cast<__gm__ dtype*>(22);                                                   \
                                                                                                                   \
        uint16_t base_idx = static_cast<uint16_t>(33);                                                             \
        uint8_t repeat = static_cast<uint8_t>(44);                                                                 \
        uint16_t src_gap = static_cast<uint16_t>(55);                                                              \
        uint8_t dst_gap = static_cast<uint8_t>(66);                                                                \
                                                                                                                   \
        asc_copy_gm2l1(dst, src, base_idx, repeat, src_gap, dst_gap);                                              \
        GlobalMockObject::verify();                                                                                \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TEST_COPY_GM_TO_L1, TEST_COPY_GM_TO_L1_SYNC_##dtype)                                                    \
    {                                                                                                              \
        MOCKER_CPP(                                                                                                \
            load_gm_to_cbuf,                                                                                       \
            void(__cbuf__ dtype*, __gm__ dtype*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, addr_cal_mode_t)) \
            .times(1)                                                                                              \
            .will(invoke(load_gm_to_cbuf_stub<dtype>));                                                            \
                                                                                                                   \
        __cbuf__ dtype* dst = reinterpret_cast<__cbuf__ dtype*>(11);                                               \
        __gm__ dtype* src = reinterpret_cast<__gm__ dtype*>(22);                                                   \
                                                                                                                   \
        uint16_t base_idx = static_cast<uint16_t>(33);                                                             \
        uint8_t repeat = static_cast<uint8_t>(44);                                                                 \
        uint16_t src_gap = static_cast<uint16_t>(55);                                                              \
        uint8_t dst_gap = static_cast<uint8_t>(66);                                                                \
                                                                                                                   \
        asc_copy_gm2l1_sync(dst, src, base_idx, repeat, src_gap, dst_gap);                                         \
        GlobalMockObject::verify();                                                                                \
    }

// ==========asc_copy_gm2l1==========
TEST_CUBE_DATAMOVE_COPY_GM2L1(bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_GM2L1(half);
TEST_CUBE_DATAMOVE_COPY_GM2L1(float);
TEST_CUBE_DATAMOVE_COPY_GM2L1(int32_t);
TEST_CUBE_DATAMOVE_COPY_GM2L1(int8_t);
TEST_CUBE_DATAMOVE_COPY_GM2L1(uint32_t);
TEST_CUBE_DATAMOVE_COPY_GM2L1(uint8_t);
