/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_STOREALIGN_INSTR_UTILS_H
#define TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_STOREALIGN_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR(class_name, c_api_name, cce_name, data_type, index)          \
                                                                                                           \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                     \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void cce_name##_##data_type##_Stub_##index(                                                            \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                     \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)           \
    {                                                                                                      \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                \
        vector_##data_type src;                                                                            \
        vector_bool mask;                                                                                  \
                                                                                                           \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                      \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                          \
                                                                                                           \
        c_api_name(dst, src, mask);                                                                        \
        GlobalMockObject::verify();                                                                        \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                 \
    protected:                                                                                         \
        void SetUp() {}                                                                                \
        void TearDown() {}                                                                             \
    };                                                                                                 \
                                                                                                       \
    namespace {                                                                                        \
    void cce_name##_##data_type##_Stub_##index(                                                        \
        vector_uint8_t src, __ubuf__ uint8_t* dst, int32_t offset, Literal dist, vector_bool mask)     \
    {}                                                                                                 \
    }                                                                                                  \
                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)       \
    {                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                            \
        vector_##data_type src;                                                                        \
        vector_bool mask;                                                                              \
                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, int32_t, Literal, vector_bool))   \
            .times(1)                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                      \
                                                                                                       \
        c_api_name(dst, src, mask);                                                                    \
        GlobalMockObject::verify();                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index)           \
                                                                                                                  \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                            \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void cce_name##_##data_type##_Stub_##index(                                                                   \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                  \
    {                                                                                                             \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                       \
        vector_int4x2_t src;                                                                                      \
        vector_bool mask;                                                                                         \
                                                                                                                  \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                             \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                 \
                                                                                                                  \
        c_api_name(dst, src, mask);                                                                               \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INSTR(class_name, c_api_name, cce_name, data_type, index)   \
                                                                                                           \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                     \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                   \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                     \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)           \
    {                                                                                                      \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                \
        vector_##data_type src;                                                                            \
        vector_bool mask;                                                                                  \
        int32_t offset;                                                                                    \
                                                                                                           \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                      \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                 \
                                                                                                           \
        c_api_name(dst, src, offset, mask);                                                                \
        GlobalMockObject::verify();                                                                        \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                              \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                        \
    protected:                                                                                                \
        void SetUp() {}                                                                                       \
        void TearDown() {}                                                                                    \
    };                                                                                                        \
                                                                                                              \
    namespace {                                                                                               \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                      \
        vector_uint8_t src, __ubuf__ uint8_t* dst, int32_t offset, Literal dist, vector_bool mask)            \
    {}                                                                                                        \
    }                                                                                                         \
                                                                                                              \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)              \
    {                                                                                                         \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                   \
        vector_##data_type src;                                                                               \
        vector_bool mask;                                                                                     \
        int32_t offset;                                                                                       \
                                                                                                              \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, int32_t, Literal, vector_bool))          \
            .times(1)                                                                                         \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                    \
                                                                                                              \
        c_api_name(dst, src, offset, mask);                                                                   \
        GlobalMockObject::verify();                                                                           \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index)    \
                                                                                                                  \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                            \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                          \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                  \
    {                                                                                                             \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                       \
        vector_int4x2_t src;                                                                                      \
        vector_bool mask;                                                                                         \
        int32_t offset;                                                                                           \
                                                                                                                  \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                             \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                        \
                                                                                                                  \
        c_api_name(dst, src, offset, mask);                                                                       \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_POSTUPDATE_INSTR(class_name, c_api_name, cce_name, data_type, index)    \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub_##index##_offset##_update(                                                      \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask, Literal mode) \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_##data_type src;                                                                                        \
        vector_bool mask;                                                                                              \
        int32_t offset;                                                                                                \
                                                                                                                       \
        c_api_name(dst, src, offset, mask);                                                                            \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR(class_name, c_api_name, cce_name, data_type, index)    \
                                                                                                           \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                     \
    protected:                                                                                             \
        void SetUp() {}                                                                                    \
        void TearDown() {}                                                                                 \
    };                                                                                                     \
                                                                                                           \
    namespace {                                                                                            \
    void cce_name##_##data_type##_Stub_##index(                                                            \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                     \
    }                                                                                                      \
                                                                                                           \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)           \
    {                                                                                                      \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                \
        vector_##data_type src;                                                                            \
                                                                                                           \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                      \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                          \
                                                                                                           \
        c_api_name(dst, src);                                                                              \
        GlobalMockObject::verify();                                                                        \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                             \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                       \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void cce_name##_##data_type##_Stub_##index(                                                              \
        vector_uint8_t src, __ubuf__ uint8_t* dst, int32_t offset, Literal dist, vector_bool mask)           \
    {}                                                                                                       \
    }                                                                                                        \
                                                                                                             \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)             \
    {                                                                                                        \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                  \
        vector_##data_type src;                                                                              \
                                                                                                             \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, int32_t, Literal, vector_bool))         \
            .times(1)                                                                                        \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                            \
                                                                                                             \
        c_api_name(dst, src);                                                                                \
        GlobalMockObject::verify();                                                                          \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index)     \
                                                                                                                  \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                            \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void cce_name##_##data_type##_Stub_##index(                                                                   \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, Literal dist, vector_bool mask)   \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                  \
    {                                                                                                             \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                       \
        vector_int4x2_t src;                                                                                      \
                                                                                                                  \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, Literal, vector_bool)) \
            .times(1)                                                                                             \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                 \
                                                                                                                  \
        c_api_name(dst, src);                                                                                     \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_OFFSET_INSTR(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                               \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                         \
    protected:                                                                                                 \
        void SetUp() {}                                                                                        \
        void TearDown() {}                                                                                     \
    };                                                                                                         \
                                                                                                               \
    namespace {                                                                                                \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                       \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask)       \
    {}                                                                                                         \
    }                                                                                                          \
                                                                                                               \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)               \
    {                                                                                                          \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                    \
        vector_##data_type src;                                                                                \
        int32_t offset;                                                                                        \
                                                                                                               \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, int32_t, Literal, vector_bool))     \
            .times(1)                                                                                          \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                     \
                                                                                                               \
        c_api_name(dst, src, offset);                                                                          \
        GlobalMockObject::verify();                                                                            \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_OFFSET_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                                    \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                              \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                            \
        vector_uint8_t src, __ubuf__ uint8_t* dst, int32_t offset, Literal dist, vector_bool mask)                  \
    {}                                                                                                              \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                    \
    {                                                                                                               \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                         \
        vector_##data_type src;                                                                                     \
        int32_t offset;                                                                                             \
                                                                                                                    \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, int32_t, Literal, vector_bool))                \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                          \
                                                                                                                    \
        c_api_name(dst, src, offset);                                                                               \
        GlobalMockObject::verify();                                                                                 \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_OFFSET_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                             \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, Literal dist, vector_bool mask)      \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_int4x2_t src;                                                                                         \
        int32_t offset;                                                                                              \
                                                                                                                     \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, Literal, vector_bool))    \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub_##index##_offset));                                           \
                                                                                                                     \
        c_api_name(dst, src, offset);                                                                                \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_OFFSET_POSTUPDATE_INSTR(                                                 \
    class_name, c_api_name, cce_name, data_type, index)                                                                \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub_##index##_offset##_update(                                                      \
        vector_##data_type src, __ubuf__ data_type* dst, int32_t offset, Literal dist, vector_bool mask, Literal mode) \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_##data_type src;                                                                                        \
        int32_t offset;                                                                                                \
                                                                                                                       \
        c_api_name(dst, src, offset);                                                                                  \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR(class_name, c_api_name, cce_name, data_type, index)             \
                                                                                                                    \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                              \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
    void cce_name##_##data_type##_Stub_##index(                                                                     \
        vector_##data_type src0, vector_##data_type src1, __ubuf__ data_type* dst, int32_t offset, int dist,        \
        vector_bool mask)                                                                                           \
    {}                                                                                                              \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                    \
    {                                                                                                               \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                         \
        vector_##data_type src0;                                                                                    \
        vector_##data_type src1;                                                                                    \
                                                                                                                    \
        MOCKER_CPP(                                                                                                 \
            cce_name, void(vector_##data_type, vector_##data_type, __ubuf__ data_type*, int32_t, int, vector_bool)) \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                   \
                                                                                                                    \
        c_api_name(dst, src0, src1);                                                                                \
        GlobalMockObject::verify();                                                                                 \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index)         \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub_##index(                                                                      \
        vector_uint8_t src0, vector_uint8_t src1, __ubuf__ uint8_t* dst, int32_t offset, int dist, vector_bool mask) \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_##data_type src0;                                                                                     \
        vector_##data_type src1;                                                                                     \
                                                                                                                     \
        MOCKER_CPP(cce_name, void(vector_uint8_t, vector_uint8_t, __ubuf__ uint8_t*, int32_t, int, vector_bool))     \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                    \
                                                                                                                     \
        c_api_name(dst, src0, src1);                                                                                 \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index)        \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub_##index(                                                                      \
        vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, int dist, \
        vector_bool mask)                                                                                            \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_int4x2_t src0;                                                                                        \
        vector_int4x2_t src1;                                                                                        \
                                                                                                                     \
        MOCKER_CPP(                                                                                                  \
            cce_name,                                                                                                \
            void(vector_fp4x2_e1m2_t, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, int, vector_bool))    \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                    \
                                                                                                                     \
        c_api_name(dst, src0, src1);                                                                                 \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INTLV_INSTR(class_name, c_api_name, cce_name, data_type, index)      \
                                                                                                                    \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                              \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                            \
        vector_##data_type src0, vector_##data_type src1, __ubuf__ data_type* dst, int32_t offset, int dist,        \
        vector_bool mask)                                                                                           \
    {}                                                                                                              \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                    \
    {                                                                                                               \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                         \
        vector_##data_type src0;                                                                                    \
        vector_##data_type src1;                                                                                    \
        int32_t offset;                                                                                             \
                                                                                                                    \
        MOCKER_CPP(                                                                                                 \
            cce_name, void(vector_##data_type, vector_##data_type, __ubuf__ data_type*, int32_t, int, vector_bool)) \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                   \
                                                                                                                    \
        c_api_name(dst, src0, src1, offset);                                                                        \
        GlobalMockObject::verify();                                                                                 \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INTLV_INSTR_HIF8(class_name, c_api_name, cce_name, data_type, index)  \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                             \
        vector_uint8_t src0, vector_uint8_t src1, __ubuf__ uint8_t* dst, int32_t offset, int dist, vector_bool mask) \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_##data_type src0;                                                                                     \
        vector_##data_type src1;                                                                                     \
        int32_t offset;                                                                                              \
                                                                                                                     \
        MOCKER_CPP(cce_name, void(vector_uint8_t, vector_uint8_t, __ubuf__ uint8_t*, int32_t, int, vector_bool))     \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                    \
                                                                                                                     \
        c_api_name(dst, src0, src1, offset);                                                                         \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INTLV_INSTR_INT4B(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                                     \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                               \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void cce_name##_##data_type##_Stub_##index##_offset(                                                             \
        vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst, int32_t offset, int dist, \
        vector_bool mask)                                                                                            \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                     \
    {                                                                                                                \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                          \
        vector_int4x2_t src0;                                                                                        \
        vector_int4x2_t src1;                                                                                        \
        int32_t offset;                                                                                              \
                                                                                                                     \
        MOCKER_CPP(                                                                                                  \
            cce_name,                                                                                                \
            void(vector_fp4x2_e1m2_t, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, int32_t, int, vector_bool))    \
            .times(1)                                                                                                \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                    \
                                                                                                                     \
        c_api_name(dst, src0, src1, offset);                                                                         \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_OFFSET_INTLV_POSTUPDATE_INSTR(                                       \
    class_name, c_api_name, cce_name, data_type, index)                                                      \
                                                                                                             \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                       \
    protected:                                                                                               \
        void SetUp() {}                                                                                      \
        void TearDown() {}                                                                                   \
    };                                                                                                       \
                                                                                                             \
    namespace {                                                                                              \
    void cce_name##_##data_type##_Stub_##index##_offset_update(                                              \
        vector_##data_type src0, vector_##data_type src1, __ubuf__ data_type* dst, int32_t offset, int dist, \
        vector_bool mask, int mode)                                                                          \
    {}                                                                                                       \
    }                                                                                                        \
                                                                                                             \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)             \
    {                                                                                                        \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                  \
        vector_##data_type src0;                                                                             \
        vector_##data_type src1;                                                                             \
        int32_t offset;                                                                                      \
                                                                                                             \
        c_api_name(dst, src0, src1, offset);                                                                 \
        GlobalMockObject::verify();                                                                          \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_1(class_name, c_api_name, cce_name, data_type, index)         \
                                                                                                            \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                      \
    protected:                                                                                              \
        void SetUp() {}                                                                                     \
        void TearDown() {}                                                                                  \
    };                                                                                                      \
                                                                                                            \
    namespace {                                                                                             \
    void cce_name##_##data_type##_Stub_##index(                                                             \
        vector_##data_type src, __ubuf__ data_type* dst, addr_reg offset, Literal dist, vector_bool mask)   \
    {}                                                                                                      \
    }                                                                                                       \
                                                                                                            \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)            \
    {                                                                                                       \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                 \
        vector_##data_type src;                                                                             \
        vector_bool mask;                                                                                   \
        addr_reg offset;                                                                                    \
                                                                                                            \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                       \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                           \
                                                                                                            \
        c_api_name(dst, src, offset, mask);                                                                 \
        GlobalMockObject::verify();                                                                         \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_1_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                         \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                   \
    protected:                                                                                           \
        void SetUp() {}                                                                                  \
        void TearDown() {}                                                                               \
    };                                                                                                   \
                                                                                                         \
    namespace {                                                                                          \
    void cce_name##_##data_type##_Stub_##index(                                                          \
        vector_uint8_t src, __ubuf__ uint8_t* dst, addr_reg offset, Literal dist, vector_bool mask)      \
    {}                                                                                                   \
    }                                                                                                    \
                                                                                                         \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)         \
    {                                                                                                    \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                              \
        vector_##data_type src;                                                                          \
        vector_bool mask;                                                                                \
        addr_reg offset;                                                                                 \
                                                                                                         \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, addr_reg, Literal, vector_bool))    \
            .times(1)                                                                                    \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                        \
                                                                                                         \
        c_api_name(dst, src, offset, mask);                                                              \
        GlobalMockObject::verify();                                                                      \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_1_INT4B(class_name, c_api_name, cce_name, data_type, index)          \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub_##index(                                                                    \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, addr_reg offset, Literal dist, vector_bool mask)   \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_int4x2_t src;                                                                                       \
        vector_bool mask;                                                                                          \
        addr_reg offset;                                                                                           \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                  \
                                                                                                                   \
        c_api_name(dst, src, offset, mask);                                                                        \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR_1(class_name, c_api_name, cce_name, data_type, index)   \
                                                                                                            \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                      \
    protected:                                                                                              \
        void SetUp() {}                                                                                     \
        void TearDown() {}                                                                                  \
    };                                                                                                      \
                                                                                                            \
    namespace {                                                                                             \
    void cce_name##_##data_type##_Stub_##index(                                                             \
        vector_##data_type src, __ubuf__ data_type* dst, addr_reg offset, Literal dist, vector_bool mask)   \
    {}                                                                                                      \
    }                                                                                                       \
                                                                                                            \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)            \
    {                                                                                                       \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                 \
        vector_##data_type src;                                                                             \
        addr_reg offset;                                                                                    \
                                                                                                            \
        MOCKER_CPP(cce_name, void(vector_##data_type, __ubuf__ data_type*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                       \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                           \
                                                                                                            \
        c_api_name(dst, src, offset);                                                                       \
        GlobalMockObject::verify();                                                                         \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR_1_HIF8(class_name, c_api_name, cce_name, data_type, index) \
                                                                                                               \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                         \
    protected:                                                                                                 \
        void SetUp() {}                                                                                        \
        void TearDown() {}                                                                                     \
    };                                                                                                         \
                                                                                                               \
    namespace {                                                                                                \
    void cce_name##_##data_type##_Stub_##index(                                                                \
        vector_uint8_t src, __ubuf__ uint8_t* dst, addr_reg offset, Literal dist, vector_bool mask)            \
    {}                                                                                                         \
    }                                                                                                          \
                                                                                                               \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)               \
    {                                                                                                          \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                    \
        vector_##data_type src;                                                                                \
        addr_reg offset;                                                                                       \
                                                                                                               \
        MOCKER_CPP(cce_name, void(vector_uint8_t, __ubuf__ uint8_t*, addr_reg, Literal, vector_bool))          \
            .times(1)                                                                                          \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                              \
                                                                                                               \
        c_api_name(dst, src, offset);                                                                          \
        GlobalMockObject::verify();                                                                            \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_ONEPT_INSTR_1_INT4B(class_name, c_api_name, cce_name, data_type, index)    \
                                                                                                                   \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                             \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void cce_name##_##data_type##_Stub_##index(                                                                    \
        vector_fp4x2_e1m2_t src, __ubuf__ float4_e1m2x2_t* dst, addr_reg offset, Literal dist, vector_bool mask)   \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                   \
    {                                                                                                              \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                        \
        vector_int4x2_t src;                                                                                       \
        addr_reg offset;                                                                                           \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                  \
                                                                                                                   \
        c_api_name(dst, src, offset);                                                                              \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR_1(class_name, c_api_name, cce_name, data_type, index)         \
                                                                                                                  \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                            \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void cce_name##_##data_type##_Stub_##index(                                                                   \
        vector_##data_type src0, vector_##data_type src1, __ubuf__ data_type* dst, addr_reg offset, Literal dist, \
        vector_bool mask)                                                                                         \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                  \
    {                                                                                                             \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                       \
        vector_##data_type src0;                                                                                  \
        vector_##data_type src1;                                                                                  \
        addr_reg offset;                                                                                          \
                                                                                                                  \
        MOCKER_CPP(                                                                                               \
            cce_name,                                                                                             \
            void(vector_##data_type, vector_##data_type, __ubuf__ data_type*, addr_reg, Literal, vector_bool))    \
            .times(1)                                                                                             \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                 \
                                                                                                                  \
        c_api_name(dst, src0, src1, offset);                                                                      \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR_1_HIF8(class_name, c_api_name, cce_name, data_type, index)        \
                                                                                                                      \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                \
    protected:                                                                                                        \
        void SetUp() {}                                                                                               \
        void TearDown() {}                                                                                            \
    };                                                                                                                \
                                                                                                                      \
    namespace {                                                                                                       \
    void cce_name##_##data_type##_Stub_##index(                                                                       \
        vector_uint8_t src0, vector_uint8_t src1, __ubuf__ uint8_t* dst, addr_reg offset, Literal dist,               \
        vector_bool mask)                                                                                             \
    {}                                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                      \
    {                                                                                                                 \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                           \
        vector_##data_type src0;                                                                                      \
        vector_##data_type src1;                                                                                      \
        addr_reg offset;                                                                                              \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(vector_uint8_t, vector_uint8_t, __ubuf__ uint8_t*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                     \
                                                                                                                      \
        c_api_name(dst, src0, src1, offset);                                                                          \
        GlobalMockObject::verify();                                                                                   \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INTLV_INSTR_1_INT4B(class_name, c_api_name, cce_name, data_type, index)        \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub_##index(                                                                        \
        vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, __ubuf__ float4_e1m2x2_t* dst, addr_reg offset,            \
        Literal dist, vector_bool mask)                                                                                \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_int4x2_t src0;                                                                                          \
        vector_int4x2_t src1;                                                                                          \
        addr_reg offset;                                                                                               \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name,                                                                                                  \
            void(vector_fp4x2_e1m2_t, vector_fp4x2_e1m2_t, __ubuf__ float4_e1m2x2_t*, addr_reg, Literal, vector_bool)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub_##index));                                                      \
                                                                                                                       \
        c_api_name(dst, src0, src1, offset);                                                                           \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_PST(data_type)                                                  \
                                                                                                              \
    class TestVectorDataMoveStorealignInstrPst##_##data_type : public testing::Test {                         \
    protected:                                                                                                \
        void SetUp() {}                                                                                       \
        void TearDown() {}                                                                                    \
    };                                                                                                        \
                                                                                                              \
    namespace {                                                                                               \
    void Stub_TestVectorDataMoveStorealignInstrPst_Norm_##data_type(                                          \
        vector_bool src, __ubuf__ uint32_t* base, vector_address offset, Literal dist)                        \
    {}                                                                                                        \
    void Stub_TestVectorDataMoveStorealignInstrPst_Pack_##data_type(                                          \
        vector_bool src, __ubuf__ uint32_t* base, vector_address offset, Literal dist)                        \
    {}                                                                                                        \
    }                                                                                                         \
                                                                                                              \
    TEST_F(TestVectorDataMoveStorealignInstrPst##_##data_type, test_normal)                                   \
    {                                                                                                         \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                   \
        vector_bool src;                                                                                      \
        vector_address offset;                                                                                \
                                                                                                              \
        MOCKER_CPP(pst, void(vector_bool src, __ubuf__ uint32_t * base, vector_address offset, Literal dist)) \
            .times(1)                                                                                         \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPst_Norm_##data_type));                        \
                                                                                                              \
        asc_storealign(dst, src, offset);                                                                     \
        GlobalMockObject::verify();                                                                           \
    }                                                                                                         \
    TEST_F(TestVectorDataMoveStorealignInstrPst##_##data_type, test_pack)                                     \
    {                                                                                                         \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                   \
        vector_bool src;                                                                                      \
        vector_address offset;                                                                                \
                                                                                                              \
        MOCKER_CPP(pst, void(vector_bool src, __ubuf__ uint32_t * base, vector_address offset, Literal dist)) \
            .times(1)                                                                                         \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPst_Pack_##data_type));                        \
                                                                                                              \
        asc_storealign_pack(dst, src, offset);                                                                \
        GlobalMockObject::verify();                                                                           \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_PSTS_1(data_type)                                         \
                                                                                                        \
    class TestVectorDataMoveStorealignInstrPsts1##_##data_type : public testing::Test {                 \
    protected:                                                                                          \
        void SetUp() {}                                                                                 \
        void TearDown() {}                                                                              \
    };                                                                                                  \
                                                                                                        \
    namespace {                                                                                         \
    void Stub_TestVectorDataMoveStorealignInstrPsts_1_Norm_##data_type(                                 \
        vector_bool src, __ubuf__ uint32_t* base, int32_t offset, Literal dist)                         \
    {}                                                                                                  \
    void Stub_TestVectorDataMoveStorealignInstrPsts_1_Pack_##data_type(                                 \
        vector_bool src, __ubuf__ uint32_t* base, int32_t offset, Literal dist)                         \
    {}                                                                                                  \
    }                                                                                                   \
                                                                                                        \
    TEST_F(TestVectorDataMoveStorealignInstrPsts1##_##data_type, test_normal)                           \
    {                                                                                                   \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                             \
        vector_bool src;                                                                                \
                                                                                                        \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * base, int32_t offset, Literal dist)) \
            .times(1)                                                                                   \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_1_Norm_##data_type));               \
                                                                                                        \
        asc_storealign(dst, src);                                                                       \
        GlobalMockObject::verify();                                                                     \
    }                                                                                                   \
    TEST_F(TestVectorDataMoveStorealignInstrPsts1##_##data_type, test_pack)                             \
    {                                                                                                   \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                             \
        vector_bool src;                                                                                \
                                                                                                        \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * base, int32_t offset, Literal dist)) \
            .times(1)                                                                                   \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_1_Pack_##data_type));               \
                                                                                                        \
        asc_storealign_pack(dst, src);                                                                  \
        GlobalMockObject::verify();                                                                     \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_PSTS_2(data_type)                                         \
                                                                                                        \
    class TestVectorDataMoveStorealignInstrPsts2##_##data_type : public testing::Test {                 \
    protected:                                                                                          \
        void SetUp() {}                                                                                 \
        void TearDown() {}                                                                              \
    };                                                                                                  \
                                                                                                        \
    namespace {                                                                                         \
    void Stub_TestVectorDataMoveStorealignInstrPsts_2_Norm_##data_type(                                 \
        vector_bool src, __ubuf__ uint32_t* base, int32_t offset, Literal dist)                         \
    {}                                                                                                  \
    void Stub_TestVectorDataMoveStorealignInstrPsts_2_Pack_##data_type(                                 \
        vector_bool src, __ubuf__ uint32_t* base, int32_t offset, Literal dist)                         \
    {}                                                                                                  \
    }                                                                                                   \
                                                                                                        \
    TEST_F(TestVectorDataMoveStorealignInstrPsts2##_##data_type, test_normal)                           \
    {                                                                                                   \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                             \
        vector_bool src;                                                                                \
                                                                                                        \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * base, int32_t offset, Literal dist)) \
            .times(1)                                                                                   \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_2_Norm_##data_type));               \
                                                                                                        \
        asc_storealign(dst, src, 0);                                                                    \
        GlobalMockObject::verify();                                                                     \
    }                                                                                                   \
    TEST_F(TestVectorDataMoveStorealignInstrPsts2##_##data_type, test_pack)                             \
    {                                                                                                   \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                             \
        vector_bool src;                                                                                \
                                                                                                        \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * base, int32_t offset, Literal dist)) \
            .times(1)                                                                                   \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_2_Pack_##data_type));               \
                                                                                                        \
        asc_storealign_pack(dst, src, 0);                                                               \
        GlobalMockObject::verify();                                                                     \
    }

#define TEST_VECTOR_DATAMOVE_STOREALIGN_INSTR_PSTS_3(data_type)                                                        \
                                                                                                                       \
    class TestVectorDataMoveStorealignInstrPsts3##_##data_type : public testing::Test {                                \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void Stub_TestVectorDataMoveStorealignInstrPsts_3_Norm_##data_type(                                                \
        vector_bool src, __ubuf__ uint32_t*& base, int32_t offset, Literal dist, Literal post)                         \
    {}                                                                                                                 \
    void Stub_TestVectorDataMoveStorealignInstrPsts_3_Pack_##data_type(                                                \
        vector_bool src, __ubuf__ uint32_t*& base, int32_t offset, Literal dist, Literal post)                         \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMoveStorealignInstrPsts3##_##data_type, test_normal)                                          \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_bool src;                                                                                               \
                                                                                                                       \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * &base, int32_t offset, Literal dist, Literal post)) \
            .times(1)                                                                                                  \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_3_Norm_##data_type));                              \
                                                                                                                       \
        asc_storealign_postupdate(dst, src, 0);                                                                        \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
    TEST_F(TestVectorDataMoveStorealignInstrPsts3##_##data_type, test_pack)                                            \
    {                                                                                                                  \
        __ubuf__ data_type* dst = reinterpret_cast<__ubuf__ data_type*>(0);                                            \
        vector_bool src;                                                                                               \
                                                                                                                       \
        MOCKER_CPP(psts, void(vector_bool src, __ubuf__ uint32_t * &base, int32_t offset, Literal dist, Literal post)) \
            .times(1)                                                                                                  \
            .will(invoke(Stub_TestVectorDataMoveStorealignInstrPsts_3_Pack_##data_type));                              \
                                                                                                                       \
        asc_storealign_pack_postupdate(dst, src, 0);                                                                   \
        GlobalMockObject::verify();                                                                                    \
    }

#endif
