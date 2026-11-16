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

#define TEST_VECTOR_COMPUTE_INTLV(data_type)                                                              \
                                                                                                          \
    class TestVectorComputeIntlv##data_type##CApi : public testing::Test {                                \
    protected:                                                                                            \
        void SetUp() {}                                                                                   \
        void TearDown() {}                                                                                \
    };                                                                                                    \
                                                                                                          \
    namespace {                                                                                           \
    void vintlv##_##data_type##_Stub(data_type& dst0, data_type& dst1, data_type src0, data_type src1) {} \
    }                                                                                                     \
                                                                                                          \
    TEST_F(TestVectorComputeIntlv##data_type##CApi, c_api_asc_intlv_##data_type##_Succ)                   \
    {                                                                                                     \
        data_type dst0;                                                                                   \
        data_type dst1;                                                                                   \
        data_type src0;                                                                                   \
        data_type src1;                                                                                   \
                                                                                                          \
        MOCKER_CPP(vintlv, void(data_type&, data_type&, data_type, data_type))                            \
            .times(1)                                                                                     \
            .will(invoke(vintlv##_##data_type##_Stub));                                                   \
                                                                                                          \
        asc_intlv(dst0, dst1, src0, src1);                                                                \
        GlobalMockObject::verify();                                                                       \
    }

#define TEST_VECTOR_COMPUTE_INTLV_HIF8(data_type)                                                  \
                                                                                                   \
    class TestVectorComputeIntlv##data_type##CApi : public testing::Test {                         \
    protected:                                                                                     \
        void SetUp() {}                                                                            \
        void TearDown() {}                                                                         \
    };                                                                                             \
                                                                                                   \
    namespace {                                                                                    \
    void vintlv##_##data_type##_Stub(                                                              \
        vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1)      \
    {}                                                                                             \
    }                                                                                              \
                                                                                                   \
    TEST_F(TestVectorComputeIntlv##data_type##CApi, c_api_asc_intlv_##data_type##_Succ)            \
    {                                                                                              \
        data_type dst0;                                                                            \
        data_type dst1;                                                                            \
        data_type src0;                                                                            \
        data_type src1;                                                                            \
                                                                                                   \
        MOCKER_CPP(vintlv, void(vector_uint8_t&, vector_uint8_t&, vector_uint8_t, vector_uint8_t)) \
            .times(1)                                                                              \
            .will(invoke(vintlv##_##data_type##_Stub));                                            \
                                                                                                   \
        asc_intlv(dst0, dst1, src0, src1);                                                         \
        GlobalMockObject::verify();                                                                \
    }

TEST_VECTOR_COMPUTE_INTLV(vector_uint8_t);
TEST_VECTOR_COMPUTE_INTLV(vector_int8_t);
TEST_VECTOR_COMPUTE_INTLV(vector_uint16_t);
TEST_VECTOR_COMPUTE_INTLV(vector_int16_t);
TEST_VECTOR_COMPUTE_INTLV(vector_uint32_t);
TEST_VECTOR_COMPUTE_INTLV(vector_int32_t);
TEST_VECTOR_COMPUTE_INTLV(vector_half);
TEST_VECTOR_COMPUTE_INTLV(vector_float);
TEST_VECTOR_COMPUTE_INTLV(vector_fp8_e4m3fn_t);
TEST_VECTOR_COMPUTE_INTLV(vector_fp8_e5m2_t);
TEST_VECTOR_COMPUTE_INTLV(vector_fp8_e8m0_t);
TEST_VECTOR_COMPUTE_INTLV(vector_bfloat16_t);
TEST_VECTOR_COMPUTE_INTLV_HIF8(vector_hifloat8_t);

class TestVectorComputePintlv : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void pintlv_b8_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}

void pintlv_b16_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}

void pintlv_b32_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}
} // namespace

TEST_F(TestVectorComputePintlv, c_api_asc_intlv_b8_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pintlv_b8, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pintlv_b8_Stub));
    asc_intlv_b8(dst0, dst1, src0, src1);
}

TEST_F(TestVectorComputePintlv, c_api_asc_intlv_b16_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pintlv_b16, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pintlv_b16_Stub));
    asc_intlv_b16(dst0, dst1, src0, src1);
}

TEST_F(TestVectorComputePintlv, c_api_asc_intlv_b32_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pintlv_b32, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pintlv_b32_Stub));
    asc_intlv_b32(dst0, dst1, src0, src1);
}
