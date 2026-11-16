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

#define TEST_VECTOR_COMPUTE_VDINTLV(data_type)                                                             \
                                                                                                           \
    class TestVectorComputeVDIntlv##data_type##CApi : public testing::Test {                               \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void vdintlv##_##data_type##_Stub(data_type& dst0, data_type& dst1, data_type src0, data_type src1) {} \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorComputeVDIntlv##data_type##CApi, c_api_asc_vdintlv_##data_type##_Succ)                \
    {                                                                                                      \
        data_type dst0;                                                                                    \
        data_type dst1;                                                                                    \
        data_type src0;                                                                                    \
        data_type src1;                                                                                    \
                                                                                                           \
        MOCKER_CPP(vdintlv, void(data_type&, data_type&, data_type, data_type))                            \
            .times(1)                                                                                      \
            .will(invoke(vdintlv##_##data_type##_Stub));                                                   \
                                                                                                           \
        asc_deintlv(dst0, dst1, src0, src1);                                                               \
        GlobalMockObject::verify();                                                                        \
    }

#define TEST_VECTOR_COMPUTE_VDINTLV_HIF8(data_type)                                                 \
                                                                                                    \
    class TestVectorComputeVDIntlv##data_type##CApi : public testing::Test {                        \
    protected:                                                                                      \
        void SetUp() {}                                                                             \
        void TearDown() {}                                                                          \
    };                                                                                              \
                                                                                                    \
    namespace {                                                                                     \
    void vdintlv##_##data_type##_Stub(                                                              \
        vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1)       \
    {}                                                                                              \
    }                                                                                               \
                                                                                                    \
    TEST_F(TestVectorComputeVDIntlv##data_type##CApi, c_api_asc_vdintlv_##data_type##_Succ)         \
    {                                                                                               \
        data_type dst0;                                                                             \
        data_type dst1;                                                                             \
        data_type src0;                                                                             \
        data_type src1;                                                                             \
                                                                                                    \
        MOCKER_CPP(vdintlv, void(vector_uint8_t&, vector_uint8_t&, vector_uint8_t, vector_uint8_t)) \
            .times(1)                                                                               \
            .will(invoke(vdintlv##_##data_type##_Stub));                                            \
                                                                                                    \
        asc_deintlv(dst0, dst1, src0, src1);                                                        \
        GlobalMockObject::verify();                                                                 \
    }

TEST_VECTOR_COMPUTE_VDINTLV(vector_int32_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_uint32_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_int16_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_uint16_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_int8_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_uint8_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_f8e4m3);
TEST_VECTOR_COMPUTE_VDINTLV(vector_f8e5m2);
TEST_VECTOR_COMPUTE_VDINTLV(vector_f8e8m0);
TEST_VECTOR_COMPUTE_VDINTLV(vector_bfloat16_t);
TEST_VECTOR_COMPUTE_VDINTLV(vector_float);
TEST_VECTOR_COMPUTE_VDINTLV(vector_half);
TEST_VECTOR_COMPUTE_VDINTLV_HIF8(vector_hifloat8_t);

class TestVectorComputePdintlv : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void pdintlv_b8_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}

void pdintlv_b16_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}

void pdintlv_b32_Stub(vector_bool&, vector_bool&, vector_bool, vector_bool) {}
} // namespace

TEST_F(TestVectorComputePdintlv, c_api_asc_dintlv_b8_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pdintlv_b8, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pdintlv_b8_Stub));
    asc_deintlv_b8(dst0, dst1, src0, src1);
}

TEST_F(TestVectorComputePdintlv, c_api_asc_dintlv_b16_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pdintlv_b16, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pdintlv_b16_Stub));
    asc_deintlv_b16(dst0, dst1, src0, src1);
}

TEST_F(TestVectorComputePdintlv, c_api_asc_dintlv_b32_Succ)
{
    vector_bool dst0;
    vector_bool dst1;
    vector_bool src0;
    vector_bool src1;
    MOCKER_CPP(pdintlv_b32, void(vector_bool&, vector_bool&, vector_bool, vector_bool))
        .times(1)
        .will(invoke(pdintlv_b32_Stub));
    asc_deintlv_b32(dst0, dst1, src0, src1);
}
