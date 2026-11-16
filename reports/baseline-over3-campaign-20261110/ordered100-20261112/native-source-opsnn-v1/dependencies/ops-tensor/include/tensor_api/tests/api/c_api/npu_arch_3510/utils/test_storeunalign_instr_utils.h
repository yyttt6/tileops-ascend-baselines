/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_STOREUNALIGN_INSTR_UTILS_H
#define TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_STOREUNALIGN_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(class_name, c_api_name, cce_name, data_type)                         \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(                                                                                \
        vector_store_unalign& src0, uint32_t offset, vector_##data_type src1, __ubuf__ data_type*& dst, Literal dist)  \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_store_unalign src0;                                                                                     \
        vector_##data_type src1;                                                                                       \
        uint32_t count;                                                                                                \
                                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, uint32_t, vector_##data_type, __ubuf__ data_type*&, Literal)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src0, src1, count);                                                                            \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0_HIF8(class_name, c_api_name, cce_name, data_type)              \
                                                                                                                 \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                           \
    protected:                                                                                                   \
        void SetUp() {}                                                                                          \
        void TearDown() {}                                                                                       \
    };                                                                                                           \
                                                                                                                 \
    namespace {                                                                                                  \
    void cce_name##_##data_type##_Stub(                                                                          \
        vector_store_unalign& src0, uint32_t offset, vector_uint8_t src1, __ubuf__ uint8_t*& dst, Literal dist)  \
    {}                                                                                                           \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                 \
    {                                                                                                            \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                      \
        vector_store_unalign src0;                                                                               \
        vector_##data_type src1;                                                                                 \
        uint32_t count;                                                                                          \
                                                                                                                 \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, uint32_t, vector_uint8_t, __ubuf__ uint8_t*&, Literal)) \
            .times(1)                                                                                            \
            .will(invoke(cce_name##_##data_type##_Stub));                                                        \
                                                                                                                 \
        c_api_name(dst, src0, src1, count);                                                                      \
        GlobalMockObject::verify();                                                                              \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0_INT4B(class_name, c_api_name, cce_name, data_type)                   \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(                                                                                \
        vector_store_unalign& src0, uint32_t offset, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t*& dst,         \
        Literal dist)                                                                                                  \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_store_unalign src0;                                                                                     \
        vector_int4x2_t src1;                                                                                          \
        uint32_t count;                                                                                                \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(vector_store_unalign&, uint32_t, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*&, Literal)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src0, src1, count);                                                                            \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(class_name, c_api_name, cce_name, data_type)                     \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub(                                                                            \
        vector_store_unalign& src0, addr_reg& offset, vector_##data_type src1, __ubuf__ data_type* dst, int dist)  \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_store_unalign src0;                                                                                 \
        vector_##data_type src1;                                                                                   \
        addr_reg count;                                                                                            \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, addr_reg&, vector_##data_type, __ubuf__ data_type*, int)) \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub));                                                          \
                                                                                                                   \
        c_api_name(dst, src0, src1, count);                                                                        \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1_HIF8(class_name, c_api_name, cce_name, data_type)          \
                                                                                                             \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                       \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void cce_name##_##data_type##_Stub(                                                                      \
        vector_store_unalign& src0, addr_reg& offset, vector_uint8_t src1, __ubuf__ uint8_t* dst, int dist)  \
    {}                                                                                                       \
    }                                                                                                        \
                                                                                                             \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)             \
    {                                                                                                        \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                  \
        vector_store_unalign src0;                                                                           \
        vector_##data_type src1;                                                                             \
        addr_reg count;                                                                                      \
                                                                                                             \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, addr_reg&, vector_uint8_t, __ubuf__ uint8_t*, int)) \
            .times(1)                                                                                        \
            .will(invoke(cce_name##_##data_type##_Stub));                                                    \
                                                                                                             \
        c_api_name(dst, src0, src1, count);                                                                  \
        GlobalMockObject::verify();                                                                          \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1_INT4B(class_name, c_api_name, cce_name, data_type)               \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub(                                                                            \
        vector_store_unalign& src0, addr_reg& offset, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst,     \
        int dist)                                                                                                  \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_store_unalign src0;                                                                                 \
        vector_int4x2_t src1;                                                                                      \
        addr_reg count;                                                                                            \
                                                                                                                   \
        MOCKER_CPP(                                                                                                \
            cce_name, void(vector_store_unalign&, addr_reg&, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int)) \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub));                                                          \
                                                                                                                   \
        c_api_name(dst, src0, src1, count);                                                                        \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(class_name, c_api_name, cce_name, data_type)               \
                                                                                                             \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                       \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void cce_name##_##data_type##_Stub0(                                                                     \
        vector_store_unalign& src0, uint32_t offset, vector_##data_type src1, __ubuf__ data_type* dst)       \
    {}                                                                                                       \
    }                                                                                                        \
                                                                                                             \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)             \
    {                                                                                                        \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                  \
        vector_store_unalign src0;                                                                           \
        vector_##data_type src1;                                                                             \
        uint32_t count;                                                                                      \
                                                                                                             \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, uint32_t, vector_##data_type, __ubuf__ data_type*)) \
            .times(1)                                                                                        \
            .will(invoke(cce_name##_##data_type##_Stub0));                                                   \
                                                                                                             \
        c_api_name(dst, src0, src1, count);                                                                  \
        GlobalMockObject::verify();                                                                          \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2_HIF8(class_name, c_api_name, cce_name, data_type)    \
                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                 \
    protected:                                                                                         \
        void SetUp() {}                                                                                \
        void TearDown() {}                                                                             \
    };                                                                                                 \
                                                                                                       \
    namespace {                                                                                        \
    void cce_name##_##data_type##_Stub0(                                                               \
        vector_store_unalign& src0, uint32_t offset, vector_uint8_t src1, __ubuf__ uint8_t* dst)       \
    {}                                                                                                 \
    }                                                                                                  \
                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)       \
    {                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                            \
        vector_store_unalign src0;                                                                     \
        vector_##data_type src1;                                                                       \
        uint32_t count;                                                                                \
                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, uint32_t, vector_uint8_t, __ubuf__ uint8_t*)) \
            .times(1)                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub0));                                             \
                                                                                                       \
        c_api_name(dst, src0, src1, count);                                                            \
        GlobalMockObject::verify();                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2_INT4B(class_name, c_api_name, cce_name, data_type)                \
                                                                                                                    \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                              \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
    void cce_name##_##data_type##_Stub0(                                                                            \
        vector_store_unalign& src0, uint32_t offset, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst)       \
    {}                                                                                                              \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                    \
    {                                                                                                               \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                         \
        vector_store_unalign src0;                                                                                  \
        vector_int4x2_t src1;                                                                                       \
        uint32_t count;                                                                                             \
                                                                                                                    \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, uint32_t, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*)) \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub0));                                                          \
                                                                                                                    \
        c_api_name(dst, src0, src1, count);                                                                         \
        GlobalMockObject::verify();                                                                                 \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(class_name, c_api_name, cce_name, data_type)              \
                                                                                                            \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                      \
    protected:                                                                                              \
        void SetUp() {}                                                                                     \
        void TearDown() {}                                                                                  \
    };                                                                                                      \
                                                                                                            \
    namespace {                                                                                             \
    void cce_name##_##data_type##_Stub(                                                                     \
        vector_store_unalign& src0, vector_##data_type src1, __ubuf__ data_type* dst, Literal dist)         \
    {}                                                                                                      \
    }                                                                                                       \
                                                                                                            \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)            \
    {                                                                                                       \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                 \
        vector_store_unalign src0;                                                                          \
        vector_##data_type src1;                                                                            \
                                                                                                            \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, vector_##data_type, __ubuf__ data_type*, Literal)) \
            .times(1)                                                                                       \
            .will(invoke(cce_name##_##data_type##_Stub));                                                   \
                                                                                                            \
        c_api_name(dst, src0, src1);                                                                        \
        GlobalMockObject::verify();                                                                         \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6_INT4B(class_name, c_api_name, cce_name, data_type)               \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub(                                                                            \
        vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst, Literal dist)         \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_store_unalign src0;                                                                                 \
        vector_int4x2_t src1;                                                                                      \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, Literal)) \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub));                                                          \
                                                                                                                   \
        c_api_name(dst, src0, src1);                                                                               \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_PSTU(class_name, c_api_name, cce_name, data_type, cce_dst_type)      \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub(vector_store_unalign& src0, vector_bool mask, __ubuf__ cce_dst_type*& dst) {} \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_store_unalign src0;                                                                                   \
        vector_bool mask;                                                                                            \
                                                                                                                     \
        MOCKER_CPP(cce_name, void(vector_store_unalign&, vector_bool, __ubuf__ cce_dst_type*&))                      \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub));                                                            \
                                                                                                                     \
        c_api_name(dst, src0, mask);                                                                                 \
        GlobalMockObject::verify();                                                                                  \
    }

#endif
