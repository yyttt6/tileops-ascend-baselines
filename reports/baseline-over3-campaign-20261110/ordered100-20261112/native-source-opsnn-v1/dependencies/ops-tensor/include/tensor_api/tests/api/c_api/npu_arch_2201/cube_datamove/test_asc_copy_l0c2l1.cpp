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

template <typename DTYPE>
__aicore__ inline void copy_matrix_cc_to_cbuf_b4_stub(
    __cbuf__ void* dst, __cc__ DTYPE* src, uint8_t sid, uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
    uint16_t srcStride, uint8_t UintFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN)
{
    EXPECT_EQ(dst, reinterpret_cast<__cbuf__ void*>(11));
    EXPECT_EQ(src, reinterpret_cast<__cc__ DTYPE*>(22));
    EXPECT_EQ(sid, static_cast<uint8_t>(0));
    EXPECT_EQ(NSize, static_cast<uint16_t>(33));
    EXPECT_EQ(MSize, static_cast<uint16_t>(44));
    EXPECT_EQ(dstStride_dst_D, static_cast<uint32_t>(55));
    EXPECT_EQ(srcStride, static_cast<uint16_t>(66));
    EXPECT_EQ(UintFlagMode, static_cast<uint8_t>(77));
    EXPECT_EQ(QuantPRE, static_cast<uint64_t>(88));
    EXPECT_EQ(ReLUPRE, static_cast<uint8_t>(99));
    EXPECT_EQ(channelSplit, static_cast<bool>(true));
    EXPECT_EQ(NZ2ND_EN, static_cast<bool>(true));
}

class TEST_COPY_L0C_TO_L1 : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

#define TEST_CUBE_DATAMOVE_COPY_L0C2L1(dtype)                                                                    \
                                                                                                                 \
    TEST_F(TEST_COPY_L0C_TO_L1, TEST_COPY_L0C_TO_L1_##dtype)                                                     \
    {                                                                                                            \
        MOCKER_CPP(                                                                                              \
            copy_matrix_cc_to_cbuf_b4, void(                                                                     \
                                           __cbuf__ void*, __cc__ dtype*, uint8_t, uint16_t, uint16_t, uint32_t, \
                                           uint16_t, uint8_t, QuantMode_t, uint8_t, bool, bool))                 \
            .times(1)                                                                                            \
            .will(invoke(copy_matrix_cc_to_cbuf_b4_stub<dtype>));                                                \
                                                                                                                 \
        __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(11);                                               \
        __cc__ dtype* src = reinterpret_cast<__cc__ dtype*>(22);                                                 \
                                                                                                                 \
        uint16_t NSize = static_cast<uint16_t>(33);                                                              \
        uint16_t MSize = static_cast<uint16_t>(44);                                                              \
        uint32_t dstStride_dst_D = static_cast<uint32_t>(55);                                                    \
        uint16_t srcStride = static_cast<uint16_t>(66);                                                          \
        uint8_t UintFlagMode = static_cast<uint8_t>(77);                                                         \
        QuantMode_t QuantPRE = static_cast<QuantMode_t>(88);                                                     \
        uint8_t ReLUPRE = static_cast<uint8_t>(99);                                                              \
        bool channelSplit = static_cast<bool>(true);                                                             \
        bool NZ2ND_EN = static_cast<bool>(true);                                                                 \
                                                                                                                 \
        asc_copy_l0c2l1(                                                                                         \
            dst, src, NSize, MSize, dstStride_dst_D, srcStride, UintFlagMode, QuantPRE, ReLUPRE, channelSplit,   \
            NZ2ND_EN);                                                                                           \
        GlobalMockObject::verify();                                                                              \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TEST_COPY_L0C_TO_L1, TEST_COPY_L0C_TO_L1_SYNC_##dtype)                                                \
    {                                                                                                            \
        MOCKER_CPP(                                                                                              \
            copy_matrix_cc_to_cbuf_b4, void(                                                                     \
                                           __cbuf__ void*, __cc__ dtype*, uint8_t, uint16_t, uint16_t, uint32_t, \
                                           uint16_t, uint8_t, QuantMode_t, uint8_t, bool, bool))                 \
            .times(1)                                                                                            \
            .will(invoke(copy_matrix_cc_to_cbuf_b4_stub<dtype>));                                                \
                                                                                                                 \
        __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(11);                                               \
        __cc__ dtype* src = reinterpret_cast<__cc__ dtype*>(22);                                                 \
                                                                                                                 \
        uint16_t NSize = static_cast<uint16_t>(33);                                                              \
        uint16_t MSize = static_cast<uint16_t>(44);                                                              \
        uint32_t dstStride_dst_D = static_cast<uint32_t>(55);                                                    \
        uint16_t srcStride = static_cast<uint16_t>(66);                                                          \
        uint8_t UintFlagMode = static_cast<uint8_t>(77);                                                         \
        QuantMode_t QuantPRE = static_cast<QuantMode_t>(88);                                                     \
        uint8_t ReLUPRE = static_cast<uint8_t>(99);                                                              \
        bool channelSplit = static_cast<bool>(true);                                                             \
        bool NZ2ND_EN = static_cast<bool>(true);                                                                 \
                                                                                                                 \
        asc_copy_l0c2l1_sync(                                                                                    \
            dst, src, NSize, MSize, dstStride_dst_D, srcStride, UintFlagMode, QuantPRE, ReLUPRE, channelSplit,   \
            NZ2ND_EN);                                                                                           \
        GlobalMockObject::verify();                                                                              \
    }

// ==========asc_copy_l12l0a==========
TEST_CUBE_DATAMOVE_COPY_L0C2L1(float);
TEST_CUBE_DATAMOVE_COPY_L0C2L1(int32_t);
