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

#define TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(class_name, c_api_name, cce_name, dst_type, src_type)                \
                                                                                                                 \
    class TestVectorCompute##c_api_name##dst_type##src_type##CApi : public testing::Test {                       \
    protected:                                                                                                   \
        void SetUp() {}                                                                                          \
        void TearDown() {}                                                                                       \
    };                                                                                                           \
                                                                                                                 \
    namespace {                                                                                                  \
    void c_api_name##cce_name##_##dst_type##_##src_type##_Stub(                                                  \
        dst_type& dst, src_type src, vector_bool mask, int round, int part, int mode)                            \
    {}                                                                                                           \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TestVectorCompute##c_api_name##dst_type##src_type##CApi, c_api_name##_##dst_type##_##src_type##_Succ) \
    {                                                                                                            \
        dst_type dst;                                                                                            \
        src_type src;                                                                                            \
        vector_bool mask;                                                                                        \
                                                                                                                 \
        MOCKER_CPP(cce_name, void(dst_type&, src_type, vector_bool, int, int, int))                              \
            .times(1)                                                                                            \
            .will(invoke(c_api_name##cce_name##_##dst_type##_##src_type##_Stub));                                \
                                                                                                                 \
        c_api_name(dst, src, mask);                                                                              \
        GlobalMockObject::verify();                                                                              \
    }

// ==========asc_float2int32_rd/ru/rz/rn/rna)==========
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rd, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rd_sat, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_ru, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_ru_sat, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rz, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rz_sat, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rn, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rn_sat, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rna, vcvt, vector_int32_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_float2int32_rna_sat, vcvt, vector_int32_t, vector_float);

// ==========asc_bfloat162e2m1x2_rd/ru/rz/rn/rna)==========
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rd, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rd_v2, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rd_v3, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rd_v4, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_ru, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_ru_v2, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_ru_v3, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_ru_v4, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rz, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rz_v2, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rz_v3, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rz_v4, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rn, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rn_v2, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rn_v3, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rn_v4, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rna, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rna_v2, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rna_v3, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_bfloat162e2m1x2_rna_v4, vcvt, vector_fp4x2_e2m1_t, vector_bfloat16_t);

// ==========asc_uint322uint16==========
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_uint322uint16_sat, vcvt, vector_uint16_t, vector_uint32_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_uint322uint16, vcvt, vector_uint16_t, vector_uint32_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_uint322uint16_sat_v2, vcvt, vector_uint16_t, vector_uint32_t);
TEST_VECTOR_COMPUTE_VCONV_SAT_INSTR(CONVSAT, asc_uint322uint16_v2, vcvt, vector_uint16_t, vector_uint32_t);
