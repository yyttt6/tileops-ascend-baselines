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

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(class_name, c_api_name, cce_name, data_type)   \
                                                                                                 \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {           \
    protected:                                                                                   \
        void SetUp() {}                                                                          \
        void TearDown() {}                                                                       \
    };                                                                                           \
                                                                                                 \
    namespace {                                                                                  \
    void cce_name##_##data_type##_Stub(                                                          \
        vector_store_unalign src, __ubuf__ data_type*& dst, int32_t offset, Literal dist)        \
    {}                                                                                           \
    }                                                                                            \
                                                                                                 \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ) \
    {                                                                                            \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                      \
        vector_store_unalign src;                                                                \
        addr_reg offset;                                                                         \
                                                                                                 \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ data_type*&, int32_t, Literal)) \
            .times(1)                                                                            \
            .will(invoke(cce_name##_##data_type##_Stub));                                        \
                                                                                                 \
        c_api_name(dst, src, offset);                                                            \
        GlobalMockObject::verify();                                                              \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_HIF8(class_name, c_api_name, cce_name, data_type)                    \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ uint8_t*& dst, int32_t offset, Literal dist) \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_store_unalign src;                                                                                      \
        addr_reg offset;                                                                                               \
                                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ uint8_t*&, int32_t, Literal))                         \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src, offset);                                                                                  \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_INT4B(class_name, c_api_name, cce_name, data_type)   \
                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                 \
    protected:                                                                                         \
        void SetUp() {}                                                                                \
        void TearDown() {}                                                                             \
    };                                                                                                 \
                                                                                                       \
    namespace {                                                                                        \
    void cce_name##_##data_type##_Stub(                                                                \
        vector_store_unalign src, __ubuf__ float4_e1m2x2_t*& dst, int32_t offset, Literal dist)        \
    {}                                                                                                 \
    }                                                                                                  \
                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)       \
    {                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                            \
        vector_store_unalign src;                                                                      \
        addr_reg offset;                                                                               \
                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ float4_e1m2x2_t*&, int32_t, Literal)) \
            .times(1)                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                              \
                                                                                                       \
        c_api_name(dst, src, offset);                                                                  \
        GlobalMockObject::verify();                                                                    \
    }

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_HIF8(Vstas, asc_storeunalign_post_postupdate, vstas, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstas, asc_storeunalign_post_postupdate, vstas, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_INT4B(Vstas, asc_storeunalign_post_postupdate, vstas, int4b_t);
