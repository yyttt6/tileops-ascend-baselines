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

template <typename DST_DTYPE, typename SRC_DTYPE>
__aicore__ inline void copy_matrix_cbuf_to_cc_stub(
    __cc__ DST_DTYPE* dst, __cbuf__ SRC_DTYPE* src, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    EXPECT_EQ(dst, reinterpret_cast<__cc__ DST_DTYPE*>(11));
    EXPECT_EQ(src, reinterpret_cast<__cbuf__ SRC_DTYPE*>(22));
    EXPECT_EQ(nBurst, static_cast<uint16_t>(33));
    EXPECT_EQ(lenBurst, static_cast<uint16_t>(44));
    EXPECT_EQ(srcGap, static_cast<uint16_t>(55));
    EXPECT_EQ(dstGap, static_cast<uint16_t>(66));
}

class TEST_COPY_L1_TO_L0C : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

#define TEST_CUBE_DATAMOVE_COPY_L12L0C(src_dtype, dst_dtype)                                      \
                                                                                                  \
    TEST_F(TEST_COPY_L1_TO_L0C, TEST_COPY_L1_TO_L0C_##src_dtype##dst_dtype)                       \
    {                                                                                             \
        MOCKER_CPP(                                                                               \
            copy_matrix_cbuf_to_cc,                                                               \
            void(__cc__ dst_dtype*, __cbuf__ src_dtype*, uint16_t, uint16_t, uint16_t, uint16_t)) \
            .times(1)                                                                             \
            .will(invoke(copy_matrix_cbuf_to_cc_stub<dst_dtype, src_dtype>));                     \
                                                                                                  \
        __cc__ dst_dtype* dst = reinterpret_cast<__cc__ dst_dtype*>(11);                          \
        __cbuf__ src_dtype* src = reinterpret_cast<__cbuf__ src_dtype*>(22);                      \
                                                                                                  \
        uint16_t nBurst = static_cast<uint16_t>(33);                                              \
        uint16_t lenBurst = static_cast<uint16_t>(44);                                            \
        uint16_t srcGap = static_cast<uint16_t>(55);                                              \
        uint16_t dstGap = static_cast<uint16_t>(66);                                              \
                                                                                                  \
        asc_copy_l12l0c(dst, src, nBurst, lenBurst, srcGap, dstGap);                              \
        GlobalMockObject::verify();                                                               \
    }                                                                                             \
                                                                                                  \
    TEST_F(TEST_COPY_L1_TO_L0C, TEST_COPY_L1_TO_L0C_SYNC_##src_dtype##dst_dtype)                  \
    {                                                                                             \
        MOCKER_CPP(                                                                               \
            copy_matrix_cbuf_to_cc,                                                               \
            void(__cc__ dst_dtype*, __cbuf__ src_dtype*, uint16_t, uint16_t, uint16_t, uint16_t)) \
            .times(1)                                                                             \
            .will(invoke(copy_matrix_cbuf_to_cc_stub<dst_dtype, src_dtype>));                     \
                                                                                                  \
        __cc__ dst_dtype* dst = reinterpret_cast<__cc__ dst_dtype*>(11);                          \
        __cbuf__ src_dtype* src = reinterpret_cast<__cbuf__ src_dtype*>(22);                      \
                                                                                                  \
        uint16_t nBurst = static_cast<uint16_t>(33);                                              \
        uint16_t lenBurst = static_cast<uint16_t>(44);                                            \
        uint16_t srcGap = static_cast<uint16_t>(55);                                              \
        uint16_t dstGap = static_cast<uint16_t>(66);                                              \
                                                                                                  \
        asc_copy_l12l0c_sync(dst, src, nBurst, lenBurst, srcGap, dstGap);                         \
        GlobalMockObject::verify();                                                               \
    }

// ==========asc_copy_l12l0a==========
TEST_CUBE_DATAMOVE_COPY_L12L0C(bfloat16_t, bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0C(half, half);
TEST_CUBE_DATAMOVE_COPY_L12L0C(float, half);
TEST_CUBE_DATAMOVE_COPY_L12L0C(float, bfloat16_t);
TEST_CUBE_DATAMOVE_COPY_L12L0C(float, float);
TEST_CUBE_DATAMOVE_COPY_L12L0C(int32_t, int32_t);
TEST_CUBE_DATAMOVE_COPY_L12L0C(uint32_t, uint32_t);
