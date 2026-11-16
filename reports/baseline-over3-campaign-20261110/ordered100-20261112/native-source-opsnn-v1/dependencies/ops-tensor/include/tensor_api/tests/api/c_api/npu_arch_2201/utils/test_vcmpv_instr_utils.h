/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_VCMPV_INSTR_UTILS_H
#define TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_VCMPV_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_VECTOR_COMPUTE_COMPARE_VCMPV_INSTR(class_name, c_api_name, cce_name, dst_type, src_type)              \
                                                                                                                   \
    class TestVectorCompute##class_name##_##dst_type##_##src_type : public testing ::Test {                        \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
                                                                                                                   \
    void cce_name##_##src_type##_##dst_type##_Stub(                                                                \
        __ubuf__ uint8_t* dst, __ubuf__ src_type* src0, __ubuf__ src_type* src1, uint8_t repeat,                   \
        uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, \
        uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)                                                    \
    {                                                                                                              \
        EXPECT_EQ(src0, reinterpret_cast<__ubuf__ src_type*>(11));                                                 \
        EXPECT_EQ(src1, reinterpret_cast<__ubuf__ src_type*>(22));                                                 \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ uint8_t*>(33));                                                   \
        EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                                \
        EXPECT_EQ(dst_block_stride, static_cast<uint8_t>(1));                                                      \
        EXPECT_EQ(src0_block_stride, static_cast<uint8_t>(1));                                                     \
        EXPECT_EQ(src1_block_stride, static_cast<uint8_t>(1));                                                     \
        EXPECT_EQ(dst_repeat_stride, static_cast<uint8_t>(8));                                                     \
        EXPECT_EQ(src0_repeat_stride, static_cast<uint8_t>(8));                                                    \
        EXPECT_EQ(src1_repeat_stride, static_cast<uint8_t>(8));                                                    \
    }                                                                                                              \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorCompute##class_name##_##dst_type##_##src_type, c_api_name##_asc_binary_config_succ)           \
    {                                                                                                              \
        __ubuf__ src_type* src0 = reinterpret_cast<__ubuf__ src_type*>(11);                                        \
        __ubuf__ src_type* src1 = reinterpret_cast<__ubuf__ src_type*>(22);                                        \
        __ubuf__ dst_type* dst = reinterpret_cast<__ubuf__ dst_type*>(33);                                         \
                                                                                                                   \
        uint8_t dst_block_stride = 1;                                                                              \
        uint8_t src0_block_stride = 1;                                                                             \
        uint8_t src1_block_stride = 1;                                                                             \
        uint8_t dst_repeat_stride = 8;                                                                             \
        uint8_t src0_repeat_stride = 8;                                                                            \
        uint8_t src1_repeat_stride = 8;                                                                            \
        uint8_t repeat = 1;                                                                                        \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(                                                                                        \
                          __ubuf__ uint8_t*, __ubuf__ src_type*, __ubuf__ src_type*, uint8_t, uint8_t, uint8_t,    \
                          uint8_t, uint8_t, uint8_t, uint8_t))                                                     \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##src_type##_##dst_type##_Stub));                                              \
                                                                                                                   \
        c_api_name(                                                                                                \
            dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride,    \
            src0_repeat_stride, src1_repeat_stride);                                                               \
        GlobalMockObject::verify();                                                                                \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorCompute##class_name##_##dst_type##_##src_type, c_api_name##_sync_binary_config_succ)          \
    {                                                                                                              \
        __ubuf__ src_type* src0 = reinterpret_cast<__ubuf__ src_type*>(11);                                        \
        __ubuf__ src_type* src1 = reinterpret_cast<__ubuf__ src_type*>(22);                                        \
        __ubuf__ dst_type* dst = reinterpret_cast<__ubuf__ dst_type*>(33);                                         \
                                                                                                                   \
        uint8_t dst_block_stride = 1;                                                                              \
        uint8_t src0_block_stride = 1;                                                                             \
        uint8_t src1_block_stride = 1;                                                                             \
        uint8_t dst_repeat_stride = 8;                                                                             \
        uint8_t src0_repeat_stride = 8;                                                                            \
        uint8_t src1_repeat_stride = 8;                                                                            \
        uint8_t repeat = 1;                                                                                        \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(                                                                                        \
                          __ubuf__ uint8_t*, __ubuf__ src_type*, __ubuf__ src_type*, uint8_t, uint8_t, uint8_t,    \
                          uint8_t, uint8_t, uint8_t, uint8_t))                                                     \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##src_type##_##dst_type##_Stub));                                              \
                                                                                                                   \
        c_api_name##_sync(                                                                                         \
            dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride,    \
            src0_repeat_stride, src1_repeat_stride);                                                               \
        GlobalMockObject::verify();                                                                                \
    }

#endif
