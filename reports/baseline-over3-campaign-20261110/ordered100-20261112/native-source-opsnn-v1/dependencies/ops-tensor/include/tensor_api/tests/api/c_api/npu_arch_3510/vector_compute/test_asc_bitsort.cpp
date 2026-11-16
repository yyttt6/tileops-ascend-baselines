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

class TestAscBitsortCAPI : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() { GlobalMockObject::verify(); }
};

#define TEST_ASC_BITSORT_REPEAT(data_type)                                                            \
    namespace {                                                                                       \
    void vbs_repeat_##data_type##_stub(                                                               \
        __ubuf__ data_type* dst, __ubuf__ data_type* src0, __ubuf__ uint32_t* src1, uint64_t config)  \
    {                                                                                                 \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ data_type*>(1));                                     \
        EXPECT_EQ(src0, reinterpret_cast<__ubuf__ data_type*>(2));                                    \
        EXPECT_EQ(src1, reinterpret_cast<__ubuf__ uint32_t*>(3));                                     \
        EXPECT_EQ(config, static_cast<uint64_t>(10) << 56);                                           \
    }                                                                                                 \
    }                                                                                                 \
                                                                                                      \
    TEST_F(TestAscBitsortCAPI, c_api_bitsort_##data_type##_repeat)                                    \
    {                                                                                                 \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(1);                           \
        __ubuf__ data_type* src0 = reinterpret_cast<__ubuf__ data_type*>(2);                          \
        __ubuf__ uint32_t* src1 = reinterpret_cast<__ubuf__ uint32_t*>(3);                            \
                                                                                                      \
        MOCKER_CPP(vbs, void(__ubuf__ data_type*, __ubuf__ data_type*, __ubuf__ uint32_t*, uint64_t)) \
            .stubs()                                                                                  \
            .will(invoke(vbs_repeat_##data_type##_stub));                                             \
                                                                                                      \
        asc_bitsort(dst, src0, src1, 10);                                                             \
    }                                                                                                 \
                                                                                                      \
    TEST_F(TestAscBitsortCAPI, c_api_bitsort_sync_##data_type##_repeat)                               \
    {                                                                                                 \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(1);                           \
        __ubuf__ data_type* src0 = reinterpret_cast<__ubuf__ data_type*>(2);                          \
        __ubuf__ uint32_t* src1 = reinterpret_cast<__ubuf__ uint32_t*>(3);                            \
                                                                                                      \
        MOCKER_CPP(vbs, void(__ubuf__ data_type*, __ubuf__ data_type*, __ubuf__ uint32_t*, uint64_t)) \
            .stubs()                                                                                  \
            .will(invoke(vbs_repeat_##data_type##_stub));                                             \
                                                                                                      \
        asc_bitsort_sync(dst, src0, src1, 10);                                                        \
    }

TEST_ASC_BITSORT_REPEAT(half)
TEST_ASC_BITSORT_REPEAT(float)
