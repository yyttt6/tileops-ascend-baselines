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

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(class_name, c_api_name, cce_name, data_type)                \
                                                                                                              \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                        \
    protected:                                                                                                \
        void SetUp() {}                                                                                       \
        void TearDown() {}                                                                                    \
    };                                                                                                        \
                                                                                                              \
    namespace {                                                                                               \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ data_type* dst, addr_reg offset) {} \
    }                                                                                                         \
                                                                                                              \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)              \
    {                                                                                                         \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                   \
        vector_store_unalign src;                                                                             \
        addr_reg offset;                                                                                      \
                                                                                                              \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ data_type*, addr_reg))                       \
            .times(1)                                                                                         \
            .will(invoke(cce_name##_##data_type##_Stub));                                                     \
                                                                                                              \
        c_api_name(dst, src, offset);                                                                         \
        GlobalMockObject::verify();                                                                           \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3_HIF8(class_name, c_api_name, cce_name, data_type)         \
                                                                                                            \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                      \
    protected:                                                                                              \
        void SetUp() {}                                                                                     \
        void TearDown() {}                                                                                  \
    };                                                                                                      \
                                                                                                            \
    namespace {                                                                                             \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ uint8_t* dst, addr_reg offset) {} \
    }                                                                                                       \
                                                                                                            \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)            \
    {                                                                                                       \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                 \
        vector_store_unalign src;                                                                           \
        addr_reg offset;                                                                                    \
                                                                                                            \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ uint8_t*, addr_reg))                       \
            .times(1)                                                                                       \
            .will(invoke(cce_name##_##data_type##_Stub));                                                   \
                                                                                                            \
        c_api_name(dst, src, offset);                                                                       \
        GlobalMockObject::verify();                                                                         \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3_INT4B(class_name, c_api_name, cce_name, data_type)                \
                                                                                                                    \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                              \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ float4_e1m2x2_t* dst, addr_reg offset) {} \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                    \
    {                                                                                                               \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                         \
        vector_store_unalign src;                                                                                   \
        addr_reg offset;                                                                                            \
                                                                                                                    \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ float4_e1m2x2_t*, addr_reg))                       \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub));                                                           \
                                                                                                                    \
        c_api_name(dst, src, offset);                                                                               \
        GlobalMockObject::verify();                                                                                 \
    }

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3_HIF8(Vsta, asc_storeunalign_post, vsta, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3(Vsta, asc_storeunalign_post, vsta, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_3_INT4B(Vsta, asc_storeunalign_post, vsta, int4b_t);

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(class_name, c_api_name, cce_name, data_type)               \
                                                                                                             \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                       \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ data_type* dst, int32_t offset) {} \
    }                                                                                                        \
                                                                                                             \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)             \
    {                                                                                                        \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                  \
        vector_store_unalign src;                                                                            \
        int32_t offset;                                                                                      \
                                                                                                             \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ data_type*, int32_t))                       \
            .times(1)                                                                                        \
            .will(invoke(cce_name##_##data_type##_Stub));                                                    \
                                                                                                             \
        c_api_name(dst, src, offset);                                                                        \
        GlobalMockObject::verify();                                                                          \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4_HIF8(class_name, c_api_name, cce_name, data_type)        \
                                                                                                           \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                     \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ uint8_t* dst, int32_t offset) {} \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)           \
    {                                                                                                      \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                \
        vector_store_unalign src;                                                                          \
        int32_t offset;                                                                                    \
                                                                                                           \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ uint8_t*, int32_t))                       \
            .times(1)                                                                                      \
            .will(invoke(cce_name##_##data_type##_Stub));                                                  \
                                                                                                           \
        c_api_name(dst, src, offset);                                                                      \
        GlobalMockObject::verify();                                                                        \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4_INT4B(class_name, c_api_name, cce_name, data_type)               \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ float4_e1m2x2_t* dst, int32_t offset) {} \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_store_unalign src;                                                                                  \
        int32_t offset;                                                                                            \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ float4_e1m2x2_t*, int32_t))                       \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub));                                                          \
                                                                                                                   \
        c_api_name(dst, src, offset);                                                                              \
        GlobalMockObject::verify();                                                                                \
    }

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4_HIF8(Vstas, asc_storeunalign_post, vstas, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4(Vstas, asc_storeunalign_post, vstas, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_4_INT4B(Vstas, asc_storeunalign_post, vstas, int4b_t);

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(class_name, c_api_name, cce_name, data_type)   \
                                                                                                 \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {           \
    protected:                                                                                   \
        void SetUp() {}                                                                          \
        void TearDown() {}                                                                       \
    };                                                                                           \
                                                                                                 \
    namespace {                                                                                  \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ data_type* dst) {}     \
    }                                                                                            \
                                                                                                 \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ) \
    {                                                                                            \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                      \
        vector_store_unalign src;                                                                \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ data_type*))                    \
            .times(1)                                                                            \
            .will(invoke(cce_name##_##data_type##_Stub));                                        \
                                                                                                 \
        c_api_name(dst, src);                                                                    \
        GlobalMockObject::verify();                                                              \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_INT4B(class_name, c_api_name, cce_name, data_type) \
                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {               \
    protected:                                                                                       \
        void SetUp() {}                                                                              \
        void TearDown() {}                                                                           \
    };                                                                                               \
                                                                                                     \
    namespace {                                                                                      \
    void cce_name##_##data_type##_Stub(vector_store_unalign src, __ubuf__ float4_e1m2x2_t* dst) {}   \
    }                                                                                                \
                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)     \
    {                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                          \
        vector_store_unalign src;                                                                    \
        MOCKER_CPP(cce_name, void(vector_store_unalign, __ubuf__ float4_e1m2x2_t*))                  \
            .times(1)                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub));                                            \
                                                                                                     \
        c_api_name(dst, src);                                                                        \
        GlobalMockObject::verify();                                                                  \
    }

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5(Vstar, asc_storeunalign_post, vstar, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_5_INT4B(Vstar, asc_storeunalign_post, vstar, int4b_t);
