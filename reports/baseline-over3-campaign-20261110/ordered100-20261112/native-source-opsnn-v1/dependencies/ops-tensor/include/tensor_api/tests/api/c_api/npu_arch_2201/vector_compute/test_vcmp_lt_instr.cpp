/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_api/npu_arch_2201/utils/test_vcmp_instr_utils.h"
#include "c_api/npu_arch_2201/utils/test_vcmpv_instr_utils.h"
#include "c_api/npu_arch_2201/utils/test_vcmpvs_instr_utils.h"

#define TEST_VECTOR_COMPUTE_VCMPVSNE_INSTR(class_name, c_api_name, cce_name, dst_type, src_type)                       \
                                                                                                                       \
    class TestVectorCompute##class_name##dst_type##_##src_type : public testing::Test {                                \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
                                                                                                                       \
    void cce_name##_##dst_type##_##src_type##_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_Stub(            \
        __ubuf__ dst_type* dst, __ubuf__ src_type* src, src_type value, uint8_t repeat, uint16_t dst_block_stride,     \
        uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)                             \
    {                                                                                                                  \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_type*>(11));                                                      \
        EXPECT_EQ(src, reinterpret_cast<__ubuf__ src_type*>(22));                                                      \
        src_type scalar = 3.3;                                                                                         \
        EXPECT_EQ(value, scalar);                                                                                      \
        EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                                    \
        EXPECT_EQ(dst_block_stride, static_cast<uint16_t>(1));                                                         \
        EXPECT_EQ(src_block_stride, static_cast<uint16_t>(1));                                                         \
        EXPECT_EQ(dst_repeat_stride, static_cast<uint16_t>(8));                                                        \
        EXPECT_EQ(src_repeat_stride, static_cast<uint16_t>(8));                                                        \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorCompute##class_name##dst_type##_##src_type, c_api_name##_half_half_half_UnaryConfig_Succ)         \
    {                                                                                                                  \
        __ubuf__ dst_type* dst = reinterpret_cast<__ubuf__ dst_type*>(11);                                             \
        __ubuf__ src_type* src = reinterpret_cast<__ubuf__ src_type*>(22);                                             \
        src_type value = 3.3;                                                                                          \
                                                                                                                       \
        uint16_t dst_block_stride = 1;                                                                                 \
        uint16_t src_block_stride = 1;                                                                                 \
        uint16_t dst_repeat_stride = 8;                                                                                \
        uint16_t src_repeat_stride = 8;                                                                                \
        uint8_t repeat = 1;                                                                                            \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name,                                                                                                  \
            void(__ubuf__ dst_type*, __ubuf__ src_type*, src_type, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))   \
            .times(1)                                                                                                  \
            .will(invoke(                                                                                              \
                cce_name##_##dst_type##_##src_type##_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_Stub));   \
                                                                                                                       \
        c_api_name(dst, src, value, repeat, dst_block_stride, src_block_stride, dst_repeat_stride, src_repeat_stride); \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorCompute##class_name##dst_type##_##src_type, c_api_name##_sync_half_half_half_int32_t_Succ)        \
    {                                                                                                                  \
        __ubuf__ dst_type* dst = reinterpret_cast<__ubuf__ dst_type*>(11);                                             \
        __ubuf__ src_type* src = reinterpret_cast<__ubuf__ src_type*>(22);                                             \
        src_type value = 3.3;                                                                                          \
        uint16_t dst_block_stride = 1;                                                                                 \
        uint16_t src_block_stride = 1;                                                                                 \
        uint16_t dst_repeat_stride = 8;                                                                                \
        uint16_t src_repeat_stride = 8;                                                                                \
        uint8_t repeat = 1;                                                                                            \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name,                                                                                                  \
            void(__ubuf__ dst_type*, __ubuf__ src_type*, src_type, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))   \
            .times(1)                                                                                                  \
            .will(invoke(                                                                                              \
                cce_name##_##dst_type##_##src_type##_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_Stub));   \
        c_api_name##_sync(                                                                                             \
            dst, src, value, repeat, dst_block_stride, src_block_stride, dst_repeat_stride, src_repeat_stride);        \
        GlobalMockObject::verify();                                                                                    \
    }

// ==========asc_lt(half/float)==========
TEST_VECTOR_COMPUTE_COMPARE_VCMP_INSTR(VCMPLT, asc_lt, vcmp_lt, half);
TEST_VECTOR_COMPUTE_COMPARE_VCMP_INSTR(VCMPLT, asc_lt, vcmp_lt, float);
// ==========asc_lt(half/float)==========
TEST_VECTOR_COMPUTE_COMPARE_VCMPV_INSTR(VCMPLT, asc_lt, vcmpv_lt, uint8_t, half);
TEST_VECTOR_COMPUTE_COMPARE_VCMPV_INSTR(VCMPLT, asc_lt, vcmpv_lt, uint8_t, float);
// ==========asc_ne_scalar(half/float)==========
TEST_VECTOR_COMPUTE_VCMPVSNE_INSTR(VCMPVSNE, asc_ne_scalar, vcmpvs_ne, uint8_t, half);
TEST_VECTOR_COMPUTE_VCMPVSNE_INSTR(VCMPVSNE, asc_ne_scalar, vcmpvs_ne, uint8_t, float);
