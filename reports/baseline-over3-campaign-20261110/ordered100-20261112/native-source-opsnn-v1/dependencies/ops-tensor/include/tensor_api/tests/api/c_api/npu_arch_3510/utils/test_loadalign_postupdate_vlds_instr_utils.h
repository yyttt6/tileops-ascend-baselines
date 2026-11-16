/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_LOADALIGN_POSTUPDATE_VLDS_INSTR_UTILS_H
#define TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_LOADALIGN_POSTUPDATE_VLDS_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(class_name, c_api_name, cce_name, dst_type, src_type)     \
                                                                                                                  \
    class TestVectorDatamove##class_name##dst_type##src_type##CApi : public testing::Test {                       \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void c_api_name##_##dst_type##_##src_type##_Stub(                                                             \
        dst_type& dst, __ubuf__ src_type*& src, int32_t offset, int load_dist, int post)                          \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDatamove##class_name##dst_type##src_type##CApi, c_api_name##_##dst_type##_##src_type##_Succ) \
    {                                                                                                             \
        dst_type dst;                                                                                             \
        __ubuf__ src_type* src;                                                                                   \
        int32_t offset;                                                                                           \
                                                                                                                  \
        MOCKER_CPP(cce_name, void(dst_type&, __ubuf__ src_type*&, int32_t, int, int))                             \
            .times(1)                                                                                             \
            .will(invoke(c_api_name##_##dst_type##_##src_type##_Stub));                                           \
                                                                                                                  \
        c_api_name(dst, src, offset);                                                                             \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(class_name, c_api_name, cce_name, dst_type, src_type) \
                                                                                                                   \
    class TestVectorDatamove##class_name##dst_type##src_type##CApi : public testing::Test {                        \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void c_api_name##_##dst_type##_##src_type##_Stub(                                                              \
        vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset, int load_dist, int post)                      \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDatamove##class_name##dst_type##src_type##CApi, c_api_name##_##dst_type##_##src_type##_Succ)  \
    {                                                                                                              \
        dst_type dst;                                                                                              \
        __ubuf__ src_type* src;                                                                                    \
        int32_t offset;                                                                                            \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_uint8_t&, __ubuf__ uint8_t*&, int32_t, int, int))                         \
            .times(1)                                                                                              \
            .will(invoke(c_api_name##_##dst_type##_##src_type##_Stub));                                            \
                                                                                                                   \
        c_api_name(dst, src, offset);                                                                              \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(class_name, c_api_name, cce_name, dst_type, src_type) \
                                                                                                                   \
    class TestVectorDatamove##class_name##dst_type##src_type##CApi : public testing::Test {                        \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
    void c_api_name##_##dst_type##_##src_type##_Stub(                                                              \
        vector_fp4x2_e1m2_t& dst, __ubuf__ float4_e1m2x2_t*& src, int32_t offset, int load_dist, int post)         \
    {}                                                                                                             \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDatamove##class_name##dst_type##src_type##CApi, c_api_name##_##dst_type##_##src_type##_Succ)  \
    {                                                                                                              \
        dst_type dst;                                                                                              \
        __ubuf__ src_type* src;                                                                                    \
        int32_t offset;                                                                                            \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_fp4x2_e1m2_t&, __ubuf__ float4_e1m2x2_t*&, int32_t, int, int))            \
            .times(1)                                                                                              \
            .will(invoke(c_api_name##_##dst_type##_##src_type##_Stub));                                            \
                                                                                                                   \
        c_api_name(dst, src, offset);                                                                              \
        GlobalMockObject::verify();                                                                                \
    }

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(class_name, c_api_name, cce_name, dst_type, src_type) \
                                                                                                                      \
    class TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi : public testing::Test {                 \
    protected:                                                                                                        \
        void SetUp() {}                                                                                               \
        void TearDown() {}                                                                                            \
    };                                                                                                                \
                                                                                                                      \
    namespace {                                                                                                       \
    void c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub(                                                    \
        dst_type& dst0, dst_type& dst1, __ubuf__ src_type*& src, int32_t offset, int load_dist, int post)             \
    {}                                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(                                                                                                           \
        TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi,                                           \
        c_api_name##_##dst_type##_##src_type##_Succ)                                                                  \
    {                                                                                                                 \
        dst_type dst0;                                                                                                \
        dst_type dst1;                                                                                                \
        __ubuf__ src_type* src;                                                                                       \
        int32_t offset;                                                                                               \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(dst_type&, dst_type&, __ubuf__ src_type*&, int32_t, int, int))                      \
            .times(1)                                                                                                 \
            .will(invoke(c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub));                                  \
                                                                                                                      \
        c_api_name(dst0, dst1, src, offset);                                                                          \
        GlobalMockObject::verify();                                                                                   \
    }

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR_HIF8(                                                \
    class_name, c_api_name, cce_name, dst_type, src_type)                                                            \
                                                                                                                     \
    class TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi : public testing::Test {                \
    protected:                                                                                                       \
        void SetUp() {}                                                                                              \
        void TearDown() {}                                                                                           \
    };                                                                                                               \
                                                                                                                     \
    namespace {                                                                                                      \
    void c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub(                                                   \
        vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t*& src, int32_t offset, int load_dist, int post) \
    {}                                                                                                               \
    }                                                                                                                \
                                                                                                                     \
    TEST_F(                                                                                                          \
        TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi,                                          \
        c_api_name##_##dst_type##_##src_type##_Succ)                                                                 \
    {                                                                                                                \
        dst_type dst0;                                                                                               \
        dst_type dst1;                                                                                               \
        __ubuf__ src_type* src;                                                                                      \
        int32_t offset;                                                                                              \
                                                                                                                     \
        MOCKER_CPP(cce_name, void(vector_uint8_t&, vector_uint8_t&, __ubuf__ uint8_t*&, int32_t, int, int))          \
            .times(1)                                                                                                \
            .will(invoke(c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub));                                 \
                                                                                                                     \
        c_api_name(dst0, dst1, src, offset);                                                                         \
        GlobalMockObject::verify();                                                                                  \
    }

#define TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR_INT4(                                                  \
    class_name, c_api_name, cce_name, dst_type, src_type)                                                              \
                                                                                                                       \
    class TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi : public testing::Test {                  \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub(                                                     \
        vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ float4_e1m2x2_t*& src, int32_t offset,          \
        int load_dist, int post)                                                                                       \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(                                                                                                            \
        TestVectorDatamove##class_name##dst_type##dst_type##src_type##CApi,                                            \
        c_api_name##_##dst_type##_##src_type##_Succ)                                                                   \
    {                                                                                                                  \
        dst_type dst0;                                                                                                 \
        dst_type dst1;                                                                                                 \
        __ubuf__ src_type* src;                                                                                        \
        int32_t offset;                                                                                                \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(vector_fp4x2_e1m2_t&, vector_fp4x2_e1m2_t&, __ubuf__ float4_e1m2x2_t*&, int32_t, int, int)) \
            .times(1)                                                                                                  \
            .will(invoke(c_api_name##_##dst_type##_##dst_type##_##src_type##_Stub));                                   \
                                                                                                                       \
        c_api_name(dst0, dst1, src, offset);                                                                           \
        GlobalMockObject::verify();                                                                                    \
    }

#endif
