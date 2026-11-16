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

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(class_name, c_api_name, cce_name, data_type)    \
                                                                                             \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {           \
    protected:                                                                               \
        void SetUp() {}                                                                      \
        void TearDown() {}                                                                   \
    };                                                                                       \
                                                                                             \
    namespace {                                                                              \
                                                                                             \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ data_type* src) {} \
    }                                                                                        \
                                                                                             \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)               \
    {                                                                                        \
        vector_load_unalign dst;                                                             \
        __ubuf__ data_type* src;                                                             \
                                                                                             \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ data_type*))                \
            .times(1)                                                                        \
            .will(invoke(cce_name##_##data_type##_Stub));                                    \
                                                                                             \
        c_api_name(dst, src);                                                                \
        GlobalMockObject::verify();                                                          \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_HIF8(class_name, c_api_name, cce_name, data_type) \
                                                                                               \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {             \
    protected:                                                                                 \
        void SetUp() {}                                                                        \
        void TearDown() {}                                                                     \
    };                                                                                         \
                                                                                               \
    namespace {                                                                                \
                                                                                               \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ uint8_t* src) {}     \
    }                                                                                          \
                                                                                               \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)                 \
    {                                                                                          \
        vector_load_unalign dst;                                                               \
        __ubuf__ data_type* src;                                                               \
                                                                                               \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ uint8_t*))                    \
            .times(1)                                                                          \
            .will(invoke(cce_name##_##data_type##_Stub));                                      \
                                                                                               \
        c_api_name(dst, src);                                                                  \
        GlobalMockObject::verify();                                                            \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_INT4(class_name, c_api_name, cce_name, data_type)     \
                                                                                                   \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {                 \
    protected:                                                                                     \
        void SetUp() {}                                                                            \
        void TearDown() {}                                                                         \
    };                                                                                             \
                                                                                                   \
    namespace {                                                                                    \
                                                                                                   \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ float4_e1m2x2_t* src) {} \
    }                                                                                              \
                                                                                                   \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)                     \
    {                                                                                              \
        vector_load_unalign dst;                                                                   \
        __ubuf__ data_type* src;                                                                   \
                                                                                                   \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ float4_e1m2x2_t*))                \
            .times(1)                                                                              \
            .will(invoke(cce_name##_##data_type##_Stub));                                          \
                                                                                                   \
        c_api_name(dst, src);                                                                      \
        GlobalMockObject::verify();                                                                \
    }

TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, int8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, uint8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_HIF8(LoadUnalignPre, asc_loadunalign_pre, vldas, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, int16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, uint16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, half);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, int32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, uint32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, float);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE(LoadUnalignPre, asc_loadunalign_pre, vldas, int64_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_INT4(LoadUnalignPre, asc_loadunalign_pre, vldas, int4b_t);

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(class_name, c_api_name, cce_name, data_type)              \
                                                                                                              \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {                            \
    protected:                                                                                                \
        void SetUp() {}                                                                                       \
        void TearDown() {}                                                                                    \
    };                                                                                                        \
                                                                                                              \
    namespace {                                                                                               \
                                                                                                              \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ data_type* src, addr_reg offset) {} \
    }                                                                                                         \
                                                                                                              \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)                                \
    {                                                                                                         \
        vector_load_unalign dst;                                                                              \
        __ubuf__ data_type* src;                                                                              \
        addr_reg offset;                                                                                      \
                                                                                                              \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ data_type*, addr_reg offset))                \
            .times(1)                                                                                         \
            .will(invoke(cce_name##_##data_type##_Stub));                                                     \
                                                                                                              \
        c_api_name(dst, src, offset);                                                                         \
        GlobalMockObject::verify();                                                                           \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET_HIF8(class_name, c_api_name, cce_name, data_type)       \
                                                                                                            \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {                          \
    protected:                                                                                              \
        void SetUp() {}                                                                                     \
        void TearDown() {}                                                                                  \
    };                                                                                                      \
                                                                                                            \
    namespace {                                                                                             \
                                                                                                            \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ uint8_t* src, addr_reg offset) {} \
    }                                                                                                       \
                                                                                                            \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)                              \
    {                                                                                                       \
        vector_load_unalign dst;                                                                            \
        __ubuf__ data_type* src;                                                                            \
        addr_reg offset;                                                                                    \
                                                                                                            \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ uint8_t*, addr_reg offset))                \
            .times(1)                                                                                       \
            .will(invoke(cce_name##_##data_type##_Stub));                                                   \
                                                                                                            \
        c_api_name(dst, src, offset);                                                                       \
        GlobalMockObject::verify();                                                                         \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET_INT4(class_name, c_api_name, cce_name, data_type)               \
                                                                                                                    \
    class TestVectorDatamove##class_name##data_type##CApi : public testing::Test {                                  \
    protected:                                                                                                      \
        void SetUp() {}                                                                                             \
        void TearDown() {}                                                                                          \
    };                                                                                                              \
                                                                                                                    \
    namespace {                                                                                                     \
                                                                                                                    \
    void cce_name##_##data_type##_Stub(vector_load_unalign& dst, __ubuf__ float4_e1m2x2_t* src, addr_reg offset) {} \
    }                                                                                                               \
                                                                                                                    \
    TEST_F(TestVectorDatamove##class_name##data_type##CApi, c_api_name##_Succ)                                      \
    {                                                                                                               \
        vector_load_unalign dst;                                                                                    \
        __ubuf__ data_type* src;                                                                                    \
        addr_reg offset;                                                                                            \
                                                                                                                    \
        MOCKER_CPP(cce_name, void(vector_load_unalign&, __ubuf__ float4_e1m2x2_t*, addr_reg offset))                \
            .times(1)                                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub));                                                           \
                                                                                                                    \
        c_api_name(dst, src, offset);                                                                               \
        GlobalMockObject::verify();                                                                                 \
    }

TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, int8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, uint8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET_HIF8(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, int16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, uint16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, half);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, int32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, uint32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, float);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, int64_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_PRE_OFFSET_INT4(LoadUnalignPreOffset, asc_loadunalign_pre, vlda, int4b_t);
