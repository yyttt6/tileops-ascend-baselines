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

#define TEST_VECTOR_COMPUTE_VCONV_SAT_PART(class_name, c_api_name, cce_name, dst_type, src_type)             \
                                                                                                             \
    class TestVectorCompute##class_name##_##c_api_name##_##dst_type##src_type##CApi : public testing::Test { \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void c_api_name##_##dst_type##_##src_type##_Stub(                                                        \
        dst_type& dst, src_type src, vector_bool mask, int round, int sat, int part, int mode)               \
    {}                                                                                                       \
    }                                                                                                        \
                                                                                                             \
    TEST_F(                                                                                                  \
        TestVectorCompute##class_name##_##c_api_name##_##dst_type##src_type##CApi,                           \
        c_api_name##_##dst_type##_##src_type##_Succ)                                                         \
    {                                                                                                        \
        dst_type dst;                                                                                        \
        src_type src;                                                                                        \
        vector_bool mask;                                                                                    \
                                                                                                             \
        MOCKER_CPP(cce_name, void(dst_type&, src_type, vector_bool, int, int, int, int))                     \
            .times(1)                                                                                        \
            .will(invoke(c_api_name##_##dst_type##_##src_type##_Stub));                                      \
                                                                                                             \
        c_api_name(dst, src, mask);                                                                          \
        GlobalMockObject::verify();                                                                          \
    }

// ==========(asc_float2hif8_rh/rna)==========
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_sat, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_sat_v2, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_v2, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_sat_v3, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_v3, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_sat_v4, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rh_v4, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_sat, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_sat_v2, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_v2, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_sat_v3, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_v3, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_sat_v4, vcvt, vector_hifloat8_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2hif8_rna_v4, vcvt, vector_hifloat8_t, vector_float);
// ==========(asc_float2e4m3_rn)==========
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_sat, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_sat_v2, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_v2, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_sat_v3, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_v3, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_sat_v4, vcvt, vector_fp8_e4m3fn_t, vector_float);
TEST_VECTOR_COMPUTE_VCONV_SAT_PART(ASCFLOATHIF, asc_float2e4m3_rn_v4, vcvt, vector_fp8_e4m3fn_t, vector_float);
