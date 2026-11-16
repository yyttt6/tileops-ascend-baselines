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

class TestCAPIAscSqueeze2201 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
template <typename DTYPE>
void vreduce_stub(
    __ubuf__ DTYPE* dst, __ubuf__ DTYPE* src0, __ubuf__ DTYPE* src1, uint8_t repeat, uint8_t dst_block_stride,
    uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride,
    uint8_t src1_repeat_stride)
{
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ DTYPE*>(11));
    EXPECT_EQ(src0, reinterpret_cast<__ubuf__ DTYPE*>(22));
    EXPECT_EQ(src1, reinterpret_cast<__ubuf__ DTYPE*>(33));
    EXPECT_EQ(repeat, static_cast<uint8_t>(44));
    EXPECT_EQ(dst_block_stride, static_cast<uint8_t>(55));
    EXPECT_EQ(src0_block_stride, static_cast<uint8_t>(66));
    EXPECT_EQ(src1_block_stride, static_cast<uint8_t>(77));
    EXPECT_EQ(dst_repeat_stride, static_cast<uint8_t>(88));
    EXPECT_EQ(src0_repeat_stride, static_cast<uint8_t>(99));
    EXPECT_EQ(src1_repeat_stride, static_cast<uint8_t>(11));
}
} // namespace

#define TEST_CAPI_ASC_SQUEEZE_2201(dtype)                                                                       \
    TEST_F(TestCAPIAscSqueeze2201, test_vreduce_##dtype)                                                        \
    {                                                                                                           \
        MOCKER_CPP(                                                                                             \
            vreduce, void(                                                                                      \
                         __ubuf__ dtype*, __ubuf__ dtype*, __ubuf__ dtype*, uint8_t, uint8_t, uint8_t, uint8_t, \
                         uint8_t, uint8_t, uint8_t))                                                            \
            .times(1)                                                                                           \
            .will(invoke(vreduce_stub<dtype>));                                                                 \
        __ubuf__ dtype* dst = reinterpret_cast<__ubuf__ dtype*>(11);                                            \
        __ubuf__ dtype* src0 = reinterpret_cast<__ubuf__ dtype*>(22);                                           \
        __ubuf__ dtype* src1 = reinterpret_cast<__ubuf__ dtype*>(33);                                           \
        uint8_t repeat = static_cast<uint8_t>(44);                                                              \
        uint8_t dst_block_stride = static_cast<uint8_t>(55);                                                    \
        uint8_t src0_block_stride = static_cast<uint8_t>(66);                                                   \
        uint8_t src1_block_stride = static_cast<uint8_t>(77);                                                   \
        uint8_t dst_repeat_stride = static_cast<uint8_t>(88);                                                   \
        uint8_t src0_repeat_stride = static_cast<uint8_t>(99);                                                  \
        uint8_t src1_repeat_stride = static_cast<uint8_t>(11);                                                  \
                                                                                                                \
        asc_squeeze(                                                                                            \
            dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, \
            src0_repeat_stride, src1_repeat_stride);                                                            \
        GlobalMockObject::verify();                                                                             \
    }

TEST_CAPI_ASC_SQUEEZE_2201(uint16_t);
TEST_CAPI_ASC_SQUEEZE_2201(uint32_t);
