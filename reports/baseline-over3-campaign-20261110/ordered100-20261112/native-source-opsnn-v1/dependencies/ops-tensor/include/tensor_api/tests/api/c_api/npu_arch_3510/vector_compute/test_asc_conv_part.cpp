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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(class_name, c_api_name, cce_name, dst_type, src_type)          \
                                                                                                                 \
    class TestVectorCompute##c_api_name##dst_type##src_type##CApi : public testing::Test {                       \
    protected:                                                                                                   \
        void SetUp() {}                                                                                          \
        void TearDown() {}                                                                                       \
    };                                                                                                           \
                                                                                                                 \
    namespace {                                                                                                  \
    void c_api_name##cce_name##_##dst_type##_##src_type##_Stub(                                                  \
        dst_type& dst, src_type src, vector_bool mask, int part, int mode)                                       \
    {}                                                                                                           \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TestVectorCompute##c_api_name##dst_type##src_type##CApi, c_api_name##_##dst_type##_##src_type##_Succ) \
    {                                                                                                            \
        dst_type dst;                                                                                            \
        src_type src;                                                                                            \
        vector_bool mask;                                                                                        \
                                                                                                                 \
        MOCKER_CPP(cce_name, void(dst_type&, src_type, vector_bool, int, int))                                   \
            .times(1)                                                                                            \
            .will(invoke(c_api_name##cce_name##_##dst_type##_##src_type##_Stub));                                \
                                                                                                                 \
        c_api_name(dst, src, mask);                                                                              \
        GlobalMockObject::verify();                                                                              \
    }

// ==========asc_uint82uint32==========
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(CONVSAT, asc_uint82uint32, vcvt, vector_uint32_t, vector_uint8_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(CONVSAT, asc_uint82uint32_v2, vcvt, vector_uint32_t, vector_uint8_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(CONVSAT, asc_uint82uint32_v3, vcvt, vector_uint32_t, vector_uint8_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(CONVSAT, asc_uint82uint32_v4, vcvt, vector_uint32_t, vector_uint8_t);

// ==========asc_e2m1x22bfloat16==========
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(CONVPART, asc_e2m1x22bfloat16, vcvt, vector_bfloat16_t, vector_fp4x2_e2m1_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(
    CONVPART, asc_e2m1x22bfloat16_v2, vcvt, vector_bfloat16_t, vector_fp4x2_e2m1_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(
    CONVPART, asc_e2m1x22bfloat16_v3, vcvt, vector_bfloat16_t, vector_fp4x2_e2m1_t);
TEST_VECTOR_COMPUTE_VCONV_PART_INSTR_PART(
    CONVPART, asc_e2m1x22bfloat16_v4, vcvt, vector_bfloat16_t, vector_fp4x2_e2m1_t);
