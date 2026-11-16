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

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(class_name, c_api_name, cce_name, data_type)                            \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(                                                                                \
        vector_##data_type& dst, vector_load_unalign& src0, __ubuf__ data_type*& src1, uint32_t inc, Literal post)     \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                           \
        vector_load_unalign src0;                                                                                      \
        vector_##data_type dst;                                                                                        \
        uint32_t count;                                                                                                \
                                                                                                                       \
        MOCKER_CPP(cce_name, void(vector_##data_type&, vector_load_unalign&, __ubuf__ data_type*&, uint32_t, Literal)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src0, src1, count);                                                                            \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_HIF8(class_name, c_api_name, cce_name, data_type)                 \
                                                                                                                 \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                           \
    protected:                                                                                                   \
        void SetUp() {}                                                                                          \
        void TearDown() {}                                                                                       \
    };                                                                                                           \
                                                                                                                 \
    namespace {                                                                                                  \
    void cce_name##_##data_type##_Stub(                                                                          \
        vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t*& src1, uint32_t inc, Literal post)     \
    {}                                                                                                           \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                 \
    {                                                                                                            \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                     \
        vector_load_unalign src0;                                                                                \
        vector_##data_type dst;                                                                                  \
        uint32_t count;                                                                                          \
                                                                                                                 \
        MOCKER_CPP(cce_name, void(vector_uint8_t&, vector_load_unalign&, __ubuf__ uint8_t*&, uint32_t, Literal)) \
            .times(1)                                                                                            \
            .will(invoke(cce_name##_##data_type##_Stub));                                                        \
                                                                                                                 \
        c_api_name(dst, src0, src1, count);                                                                      \
        GlobalMockObject::verify();                                                                              \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_INT4(class_name, c_api_name, cce_name, data_type)                       \
                                                                                                                       \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                                 \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(                                                                                \
        vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ float4_e1m2x2_t*& src1, uint32_t inc,            \
        Literal post)                                                                                                  \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                       \
    {                                                                                                                  \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                           \
        vector_load_unalign src0;                                                                                      \
        vector_int4x2_t dst;                                                                                           \
        uint32_t count;                                                                                                \
                                                                                                                       \
        MOCKER_CPP(                                                                                                    \
            cce_name, void(vector_fp4x2_e1m2_t&, vector_load_unalign&, __ubuf__ float4_e1m2x2_t*&, uint32_t, Literal)) \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src0, src1, count);                                                                            \
        GlobalMockObject::verify();                                                                                    \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(class_name, c_api_name, cce_name, data_type)                 \
                                                                                                                 \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                           \
    protected:                                                                                                   \
        void SetUp() {}                                                                                          \
        void TearDown() {}                                                                                       \
    };                                                                                                           \
                                                                                                                 \
    namespace {                                                                                                  \
    void cce_name##_##data_type##_Stub(                                                                          \
        vector_##data_type& dst, vector_load_unalign& src0, addr_reg& offset, __ubuf__ data_type*& src1,         \
        uint32_t post)                                                                                           \
    {}                                                                                                           \
    }                                                                                                            \
                                                                                                                 \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                 \
    {                                                                                                            \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                     \
        vector_load_unalign src0;                                                                                \
        vector_##data_type dst;                                                                                  \
        uint32_t count;                                                                                          \
        addr_reg offset;                                                                                         \
                                                                                                                 \
        MOCKER_CPP(                                                                                              \
            cce_name, void(vector_##data_type&, vector_load_unalign&, addr_reg&, __ubuf__ data_type*, uint32_t)) \
            .times(1)                                                                                            \
            .will(invoke(cce_name##_##data_type##_Stub));                                                        \
                                                                                                                 \
        c_api_name(dst, src0, src1, offset, count);                                                              \
        GlobalMockObject::verify();                                                                              \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR_HIF8(class_name, c_api_name, cce_name, data_type)             \
                                                                                                                  \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                            \
    protected:                                                                                                    \
        void SetUp() {}                                                                                           \
        void TearDown() {}                                                                                        \
    };                                                                                                            \
                                                                                                                  \
    namespace {                                                                                                   \
    void cce_name##_##data_type##_Stub(                                                                           \
        vector_uint8_t& dst, vector_load_unalign& src0, addr_reg& offset, __ubuf__ uint8_t*& src1, uint32_t post) \
    {}                                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                  \
    {                                                                                                             \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                      \
        vector_load_unalign src0;                                                                                 \
        vector_##data_type dst;                                                                                   \
        uint32_t count;                                                                                           \
        addr_reg offset;                                                                                          \
                                                                                                                  \
        MOCKER_CPP(cce_name, void(vector_uint8_t&, vector_load_unalign&, addr_reg&, __ubuf__ uint8_t*, uint32_t)) \
            .times(1)                                                                                             \
            .will(invoke(cce_name##_##data_type##_Stub));                                                         \
                                                                                                                  \
        c_api_name(dst, src0, src1, offset, count);                                                               \
        GlobalMockObject::verify();                                                                               \
    }

#define TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR_INT4(class_name, c_api_name, cce_name, data_type)           \
                                                                                                                \
    class TestVectorDataMove##class_name##_##data_type##_CApi : public testing::Test {                          \
    protected:                                                                                                  \
        void SetUp() {}                                                                                         \
        void TearDown() {}                                                                                      \
    };                                                                                                          \
                                                                                                                \
    namespace {                                                                                                 \
    void cce_name##_##data_type##_Stub(                                                                         \
        vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, addr_reg& offset, __ubuf__ float4_e1m2x2_t*& src1, \
        uint32_t post)                                                                                          \
    {}                                                                                                          \
    }                                                                                                           \
                                                                                                                \
    TEST_F(TestVectorDataMove##class_name##_##data_type##_CApi, c_api_name##_##data_type##_Succ)                \
    {                                                                                                           \
        __ubuf__ data_type* src1 = reinterpret_cast<__ubuf__ data_type*>(0);                                    \
        vector_load_unalign src0;                                                                               \
        vector_int4x2_t dst;                                                                                    \
        uint32_t count;                                                                                         \
        addr_reg offset;                                                                                        \
                                                                                                                \
        MOCKER_CPP(                                                                                             \
            cce_name,                                                                                           \
            void(vector_fp4x2_e1m2_t&, vector_load_unalign&, addr_reg&, __ubuf__ float4_e1m2x2_t*, uint32_t))   \
            .times(1)                                                                                           \
            .will(invoke(cce_name##_##data_type##_Stub));                                                       \
                                                                                                                \
        c_api_name(dst, src0, src1, offset, count);                                                             \
        GlobalMockObject::verify();                                                                             \
    }

// ==========asc_loadunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e2m1/e1m2)=========
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, int8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, uint8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, int16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, uint16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, half);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, float);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, int32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, uint32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, int64_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_HIF8(LOADUALIGN, asc_loadunalign_postupdate, vldus, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR(LOADUALIGN, asc_loadunalign_postupdate, vldus, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_INSTR_INT4(LOADUALIGN, asc_loadunalign_postupdate, vldus, int4b_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, int8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, uint8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, int16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, uint16_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, half);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, float);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, int32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, uint32_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, int64_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR_HIF8(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADUNALIGN_VLDU_INSTR_INT4(LOADUALIGNVLDU, asc_loadunalign_postupdate, vldu, int4b_t);
