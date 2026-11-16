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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_COMPUTE_LOADALIGNV2(dst_type, src_type, cce_dst_type, cce_src_type)                                \
                                                                                                                       \
    class TestVectorDataMoveLoadAlignV2##dst_type##CApi : public testing::Test {                                       \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void vsldb##_##dst_type##_Stub(cce_dst_type& dst, __ubuf__ cce_src_type* src, int32_t offset, vector_bool mask) {} \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMoveLoadAlignV2##dst_type##CApi, c_api_asc_loadalign_v2_##dst_type##_Succ)                    \
    {                                                                                                                  \
        dst_type dst;                                                                                                  \
        __ubuf__ src_type* src = reinterpret_cast<__ubuf__ src_type*>(11);                                             \
        vector_bool mask;                                                                                              \
                                                                                                                       \
        MOCKER_CPP(vsldb, void(cce_dst_type&, __ubuf__ cce_src_type*, int32_t, vector_bool))                           \
            .times(1)                                                                                                  \
            .will(invoke(vsldb##_##dst_type##_Stub));                                                                  \
                                                                                                                       \
        asc_loadalign(dst, src, 0, 0, mask);                                                                           \
        GlobalMockObject::verify();                                                                                    \
    }

TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_int8_t, int8_t, vector_int8_t, int8_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_uint8_t, uint8_t, vector_uint8_t, uint8_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_int16_t, int16_t, vector_int16_t, int16_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_uint16_t, uint16_t, vector_uint16_t, uint16_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_int32_t, int32_t, vector_int32_t, int32_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_uint32_t, uint32_t, vector_uint32_t, uint32_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_int64_t, int64_t, vector_int64_t, int64_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_bfloat16_t, bfloat16_t, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_fp8_e4m3fn_t, fp8_e4m3fn_t, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_hifloat8_t, hifloat8_t, vector_uint8_t, uint8_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_fp8_e5m2_t, fp8_e5m2_t, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_fp8_e8m0_t, fp8_e8m0_t, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_fp4x2_e1m2_t, fp4x2_e1m2_t, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_fp4x2_e2m1_t, fp4x2_e2m1_t, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_half, half, vector_half, half);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_float, float, vector_float, float);
TEST_VECTOR_COMPUTE_LOADALIGNV2(vector_int4x2_t, int4b_t, vector_fp4x2_e1m2_t, float4_e1m2x2_t);

#define TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(dst_type, src_type, cce_dst_type, cce_src_type)            \
                                                                                                              \
    class TestVectorDataMoveLoadAlignPostUpdateV2##dst_type##CApi : public testing::Test {                    \
    protected:                                                                                                \
        void SetUp() {}                                                                                       \
        void TearDown() {}                                                                                    \
    };                                                                                                        \
                                                                                                              \
    namespace {                                                                                               \
    void vsldb_post##_##dst_type##_Stub(                                                                      \
        cce_dst_type& dst, __ubuf__ cce_src_type*& src, int32_t offset, vector_bool mask, Literal mode)       \
    {}                                                                                                        \
    }                                                                                                         \
                                                                                                              \
    TEST_F(TestVectorDataMoveLoadAlignPostUpdateV2##dst_type##CApi, c_api_asc_loadalign_v2_##dst_type##_Succ) \
    {                                                                                                         \
        dst_type dst;                                                                                         \
        __ubuf__ src_type* src = reinterpret_cast<__ubuf__ src_type*>(11);                                    \
        vector_bool mask;                                                                                     \
                                                                                                              \
        MOCKER_CPP(vsldb, void(cce_dst_type&, __ubuf__ cce_src_type*&, int32_t, vector_bool, Literal))        \
            .times(1)                                                                                         \
            .will(invoke(vsldb_post##_##dst_type##_Stub));                                                    \
                                                                                                              \
        asc_loadalign_postupdate(dst, src, 0, 0, mask);                                                       \
        GlobalMockObject::verify();                                                                           \
    }

TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_int8_t, int8_t, vector_int8_t, int8_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_uint8_t, uint8_t, vector_uint8_t, uint8_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_int16_t, int16_t, vector_int16_t, int16_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_uint16_t, uint16_t, vector_uint16_t, uint16_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_int32_t, int32_t, vector_int32_t, int32_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_uint32_t, uint32_t, vector_uint32_t, uint32_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_int64_t, int64_t, vector_int64_t, int64_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_bfloat16_t, bfloat16_t, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_fp8_e4m3fn_t, fp8_e4m3fn_t, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_hifloat8_t, hifloat8_t, vector_uint8_t, uint8_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_fp8_e5m2_t, fp8_e5m2_t, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_fp8_e8m0_t, fp8_e8m0_t, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_fp4x2_e1m2_t, fp4x2_e1m2_t, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_fp4x2_e2m1_t, fp4x2_e2m1_t, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_half, half, vector_half, half);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_float, float, vector_float, float);
TEST_VECTOR_COMPUTE_LOADALIGN_POSTUPDATEV2(vector_int4x2_t, int4b_t, vector_fp4x2_e1m2_t, float4_e1m2x2_t);
