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
__aicore__ inline void img2colv2_cbuf_to_cb_stub(
    __cb__ DTYPE* dst, __cbuf__ DTYPE* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k, uint16_t pos_m,
    uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w, uint8_t dilation_h, bool filter_w,
    bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    EXPECT_EQ(dst, reinterpret_cast<__cb__ DTYPE*>(11));
    EXPECT_EQ(src, reinterpret_cast<__cbuf__ DTYPE*>(22));
    EXPECT_EQ(step_k, static_cast<uint16_t>(33));
    EXPECT_EQ(step_m, static_cast<uint16_t>(44));
    EXPECT_EQ(pos_k, static_cast<uint16_t>(55));
    EXPECT_EQ(pos_m, static_cast<uint16_t>(66));
    EXPECT_EQ(stride_w, static_cast<uint8_t>(77));
    EXPECT_EQ(stride_h, static_cast<uint8_t>(88));
    EXPECT_EQ(w_k, static_cast<uint8_t>(99));
    EXPECT_EQ(h_k, static_cast<uint8_t>(12));
    EXPECT_EQ(dilation_w, static_cast<uint8_t>(13));
    EXPECT_EQ(dilation_h, static_cast<uint8_t>(14));
    EXPECT_EQ(filter_w, static_cast<bool>(true));
    EXPECT_EQ(filter_h, static_cast<bool>(true));
    EXPECT_EQ(transpose, static_cast<bool>(true));
    EXPECT_EQ(fmatrix_ctrl, static_cast<bool>(true));
    EXPECT_EQ(size_channel, static_cast<uint16_t>(15));
}

class TEST_COPY_L1_TO_L0B : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

#define TEST_CUBE_DATAMOVE_COPY_L12L0B(dtype)                                                                   \
                                                                                                                \
    TEST_F(TEST_COPY_L1_TO_L0B, TEST_COPY_L1_TO_L0B_##dtype)                                                    \
    {                                                                                                           \
        __cb__ dtype* dst = reinterpret_cast<__cb__ dtype*>(11);                                                \
        __cbuf__ dtype* src = reinterpret_cast<__cbuf__ dtype*>(22);                                            \
                                                                                                                \
        uint16_t k_extension = static_cast<uint16_t>(33);                                                       \
        uint16_t m_extension = static_cast<uint16_t>(44);                                                       \
        uint16_t k_start_pt = static_cast<uint16_t>(55);                                                        \
        uint16_t m_start_pt = static_cast<uint16_t>(66);                                                        \
        uint8_t stride_w = static_cast<uint8_t>(77);                                                            \
        uint8_t stride_h = static_cast<uint8_t>(88);                                                            \
        uint8_t filter_w = static_cast<uint8_t>(99);                                                            \
        uint8_t filter_h = static_cast<uint8_t>(12);                                                            \
        uint8_t dilation_filter_w = static_cast<uint8_t>(13);                                                   \
        uint8_t dilation_filter_h = static_cast<uint8_t>(14);                                                   \
        bool filter_size_w = static_cast<bool>(true);                                                           \
        bool filter_size_h = static_cast<bool>(true);                                                           \
        bool transpose = static_cast<bool>(true);                                                               \
        bool fmatrix_ctrl = static_cast<bool>(true);                                                            \
        uint16_t channel_size = static_cast<uint16_t>(15);                                                      \
                                                                                                                \
        asc_copy_l12l0b(                                                                                        \
            dst, src, k_extension, m_extension, k_start_pt, m_start_pt, stride_w, stride_h, filter_w, filter_h, \
            dilation_filter_w, dilation_filter_h, filter_size_w, filter_size_h, transpose, fmatrix_ctrl,        \
            channel_size);                                                                                      \
        GlobalMockObject::verify();                                                                             \
    }                                                                                                           \
                                                                                                                \
    TEST_F(TEST_COPY_L1_TO_L0B, TEST_COPY_L1_TO_L0B_SYNC_##dtype)                                               \
    {                                                                                                           \
        __cb__ dtype* dst = reinterpret_cast<__cb__ dtype*>(11);                                                \
        __cbuf__ dtype* src = reinterpret_cast<__cbuf__ dtype*>(22);                                            \
                                                                                                                \
        uint16_t k_extension = static_cast<uint16_t>(33);                                                       \
        uint16_t m_extension = static_cast<uint16_t>(44);                                                       \
        uint16_t k_start_pt = static_cast<uint16_t>(55);                                                        \
        uint16_t m_start_pt = static_cast<uint16_t>(66);                                                        \
        uint8_t stride_w = static_cast<uint8_t>(77);                                                            \
        uint8_t stride_h = static_cast<uint8_t>(88);                                                            \
        uint8_t filter_w = static_cast<uint8_t>(99);                                                            \
        uint8_t filter_h = static_cast<uint8_t>(12);                                                            \
        uint8_t dilation_filter_w = static_cast<uint8_t>(13);                                                   \
        uint8_t dilation_filter_h = static_cast<uint8_t>(14);                                                   \
        bool filter_size_w = static_cast<bool>(true);                                                           \
        bool filter_size_h = static_cast<bool>(true);                                                           \
        bool transpose = static_cast<bool>(true);                                                               \
        bool fmatrix_ctrl = static_cast<bool>(true);                                                            \
        uint16_t channel_size = static_cast<uint16_t>(15);                                                      \
                                                                                                                \
        asc_copy_l12l0b_sync(                                                                                   \
            dst, src, k_extension, m_extension, k_start_pt, m_start_pt, stride_w, stride_h, filter_w, filter_h, \
            dilation_filter_w, dilation_filter_h, filter_size_w, filter_size_h, transpose, fmatrix_ctrl,        \
            channel_size);                                                                                      \
        GlobalMockObject::verify();                                                                             \
    }

// ==========asc_copy_l12l0b==========
TEST_CUBE_DATAMOVE_COPY_L12L0B(bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(float8_e4m3_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(float8_e5m2_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(half);
TEST_CUBE_DATAMOVE_COPY_L12L0B(float);
TEST_CUBE_DATAMOVE_COPY_L12L0B(hifloat8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(int16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(int32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(int8_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(uint16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(uint32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0B(uint8_t);
