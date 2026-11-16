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

#define TEST_VECTOR_COMPUTE_MRGSORT4_INSTR_3510(class_name, c_api_name, cce_name, data_type)                           \
                                                                                                                       \
    class TestVectorCompute3510##class_name##data_type : public testing::Test {                                        \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void c_api_name##_##data_type##_Stub(                                                                              \
        __ubuf__ data_type* dst, __ubuf__ data_type** src, uint8_t repeat, uint16_t element_length_0,                  \
        uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, \
        uint8_t valid_bit)                                                                                             \
                                                                                                                       \
    {                                                                                                                  \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ data_type*>(11));                                                     \
        EXPECT_EQ(src[0], reinterpret_cast<__ubuf__ data_type*>(22));                                                  \
        EXPECT_EQ(src[1], reinterpret_cast<__ubuf__ data_type*>(33));                                                  \
        EXPECT_EQ(src[2], reinterpret_cast<__ubuf__ data_type*>(44));                                                  \
        EXPECT_EQ(src[3], reinterpret_cast<__ubuf__ data_type*>(55));                                                  \
        EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                                    \
        EXPECT_EQ(element_length_0, static_cast<uint16_t>(2));                                                         \
        EXPECT_EQ(element_length_1, static_cast<uint16_t>(3));                                                         \
        EXPECT_EQ(element_length_2, static_cast<uint16_t>(4));                                                         \
        EXPECT_EQ(element_length_3, static_cast<uint16_t>(5));                                                         \
        EXPECT_EQ(if_exhausted_suspension, static_cast<bool>(0));                                                      \
        EXPECT_EQ(valid_bit, static_cast<uint8_t>(6));                                                                 \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorCompute3510##class_name##data_type, c_api_name##_##data_type##_Succ)                              \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(11);                                           \
        __ubuf__ data_type* src[ASC_C_API_MRGSORT_ELEMENT_LEN];                                                        \
        src[0] = reinterpret_cast<__ubuf__ data_type*>(22);                                                            \
        src[1] = reinterpret_cast<__ubuf__ data_type*>(33);                                                            \
        src[2] = reinterpret_cast<__ubuf__ data_type*>(44);                                                            \
        src[3] = reinterpret_cast<__ubuf__ data_type*>(55);                                                            \
                                                                                                                       \
        uint8_t repeat = 1;                                                                                            \
        uint16_t element_length_0 = 2;                                                                                 \
        uint16_t element_length_1 = 3;                                                                                 \
        uint16_t element_length_2 = 4;                                                                                 \
        uint16_t element_length_3 = 5;                                                                                 \
        bool if_exhausted_suspension = 0;                                                                              \
        uint8_t valid_bit = 6;                                                                                         \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(                                                                                            \
                          __ubuf__ data_type*, __ubuf__ data_type**, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t,  \
                          bool, uint8_t))                                                                              \
            .times(1)                                                                                                  \
            .will(invoke(c_api_name##_##data_type##_Stub));                                                            \
                                                                                                                       \
        c_api_name(                                                                                                    \
            dst, src, repeat, element_length_0, element_length_1, element_length_2, element_length_3,                  \
            if_exhausted_suspension, valid_bit);                                                                       \
        GlobalMockObject::verify();                                                                                    \
    }

// ==========asc_mrgsort4(half/float)==========
TEST_VECTOR_COMPUTE_MRGSORT4_INSTR_3510(Mrgsort4, asc_mrgsort4, vmrgsort4, half);
TEST_VECTOR_COMPUTE_MRGSORT4_INSTR_3510(Mrgsort4Sync, asc_mrgsort4_sync, vmrgsort4, half);
TEST_VECTOR_COMPUTE_MRGSORT4_INSTR_3510(Mrgsort4, asc_mrgsort4, vmrgsort4, float);
TEST_VECTOR_COMPUTE_MRGSORT4_INSTR_3510(Mrgsort4Sync, asc_mrgsort4_sync, vmrgsort4, float);
