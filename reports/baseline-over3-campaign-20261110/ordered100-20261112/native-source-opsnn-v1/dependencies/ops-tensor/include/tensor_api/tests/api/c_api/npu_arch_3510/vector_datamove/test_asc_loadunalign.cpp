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

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(class_name, c_api_name, cce_name, data_type)                           \
                                                                                                                      \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                \
    protected:                                                                                                        \
        void SetUp() {}                                                                                               \
        void TearDown() {}                                                                                            \
    };                                                                                                                \
                                                                                                                      \
    namespace {                                                                                                       \
    void cce_name##_##data_type##_Stub0(vector_##data_type& dst, vector_load_unalign& src0, __ubuf__ data_type* src1) \
    {}                                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                      \
    {                                                                                                                 \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_store_unalign src0;                                                                                    \
        vector_##data_type dst;                                                                                       \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(vector_##data_type&, vector_store_unalign&, __ubuf__ data_type*))                   \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##data_type##_Stub0));                                                            \
                                                                                                                      \
        c_api_name(dst, src0, src1);                                                                                  \
        GlobalMockObject::verify();                                                                                   \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_HIF8(class_name, c_api_name, cce_name, data_type)                   \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub0(vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t* src1) {} \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                       \
        vector_store_unalign src0;                                                                                 \
        vector_##data_type dst;                                                                                    \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_uint8_t&, vector_store_unalign&, __ubuf__ uint8_t*))                      \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub0));                                                         \
                                                                                                                   \
        c_api_name(dst, src0, src1);                                                                               \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_INT4(class_name, c_api_name, cce_name, data_type)           \
                                                                                                           \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                     \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void cce_name##_##data_type##_Stub0(                                                                   \
        vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ float4_e1m2x2_t* src1)               \
    {}                                                                                                     \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)           \
    {                                                                                                      \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                               \
        vector_store_unalign src0;                                                                         \
        vector_int4x2_t dst;                                                                               \
                                                                                                           \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t&, vector_store_unalign&, __ubuf__ float4_e1m2x2_t*)) \
            .times(1)                                                                                      \
            .will(invoke(cce_name##_##data_type##_Stub0));                                                 \
                                                                                                           \
        c_api_name(dst, src0, src1);                                                                       \
        GlobalMockObject::verify();                                                                        \
    }

TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, int8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, uint8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, int16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, uint16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, int32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, uint32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, int64_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, half);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, float);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_HIF8(Vldus0, asc_loadunalign, vldus, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(Vldus0, asc_loadunalign, vldus, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_INT4(Vldus0, asc_loadunalign, vldus, int4b_t);
