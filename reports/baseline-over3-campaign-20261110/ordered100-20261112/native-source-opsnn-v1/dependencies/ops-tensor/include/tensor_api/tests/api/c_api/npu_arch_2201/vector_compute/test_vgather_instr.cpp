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
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_VECTOR_COMPUTE_VGATHER_INSTR(class_name, c_api_name, cce_name, src_data_type, dst_data_type)             \
                                                                                                                      \
    class TestVectorCompute##class_name##src_data_type##dst_data_type : public testing::Test {                        \
    protected:                                                                                                        \
        void SetUp() {}                                                                                               \
        void TearDown() {}                                                                                            \
    };                                                                                                                \
                                                                                                                      \
    namespace {                                                                                                       \
                                                                                                                      \
    void cce_name##_##src_data_type##_##dst_data_type##_uint32_t_uint32_uint16_uint8_Stub(                            \
        __ubuf__ dst_data_type* dst, __ubuf__ uint32_t* src_offset, uint32_t offset_addr, uint16_t dst_repeat_stride, \
        uint8_t repeat)                                                                                               \
    {                                                                                                                 \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type*>(11));                                                \
        EXPECT_EQ(src_offset, reinterpret_cast<__ubuf__ uint32_t*>(33));                                              \
        EXPECT_EQ(offset_addr, static_cast<uint32_t>(22));                                                            \
        EXPECT_EQ(dst_repeat_stride, static_cast<uint16_t>(8));                                                       \
        EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                                   \
    }                                                                                                                 \
                                                                                                                      \
    void cce_name##_##src_data_type##_##dst_data_type##_uint32_t_uint64_t_Stub(                                       \
        __ubuf__ dst_data_type* dst, __ubuf__ uint32_t* src_offset, uint32_t offset_addr, uint16_t dst_repeat_stride, \
        uint8_t repeat)                                                                                               \
    {                                                                                                                 \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type*>(11));                                                \
        EXPECT_EQ(src_offset, reinterpret_cast<__ubuf__ uint32_t*>(33));                                              \
        EXPECT_EQ(offset_addr, static_cast<uint32_t>(22));                                                            \
    }                                                                                                                 \
                                                                                                                      \
    void cce_name##_##src_data_type##_##dst_data_type##_set_vector_mask_Stub(uint64_t mask1, uint64_t mask0)          \
    {                                                                                                                 \
        EXPECT_EQ(mask1, static_cast<uint64_t>(0));                                                                   \
        EXPECT_EQ(mask0, static_cast<uint64_t>(44));                                                                  \
    }                                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorCompute##class_name##src_data_type##dst_data_type, c_api_name##_GatherConfig_Succ)               \
    {                                                                                                                 \
        __ubuf__ src_data_type* dst = reinterpret_cast<__ubuf__ src_data_type*>(11);                                  \
        __ubuf__ src_data_type* src = reinterpret_cast<__ubuf__ src_data_type*>(22);                                  \
        __ubuf__ uint32_t* src_offset = reinterpret_cast<__ubuf__ uint32_t*>(33);                                     \
                                                                                                                      \
        uint16_t dst_repeat_stride = static_cast<uint16_t>(8);                                                        \
        uint8_t repeat = static_cast<uint8_t>(1);                                                                     \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type*, __ubuf__ uint32_t*, uint32_t, uint16_t, uint8_t))          \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##src_data_type##_##dst_data_type##_uint32_t_uint32_uint16_uint8_Stub));          \
                                                                                                                      \
        c_api_name(dst, src, src_offset, dst_repeat_stride, repeat);                                                  \
        GlobalMockObject::verify();                                                                                   \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorCompute##class_name##src_data_type##dst_data_type, c_api_name##_uint32_t_uint32_t_Succ)          \
    {                                                                                                                 \
        __ubuf__ src_data_type* dst = reinterpret_cast<__ubuf__ src_data_type*>(11);                                  \
        __ubuf__ src_data_type* src = reinterpret_cast<__ubuf__ src_data_type*>(22);                                  \
        __ubuf__ uint32_t* src_offset = reinterpret_cast<__ubuf__ uint32_t*>(33);                                     \
        uint32_t count = static_cast<uint32_t>(44);                                                                   \
        MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                                         \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##src_data_type##_##dst_data_type##_set_vector_mask_Stub));                       \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type*, __ubuf__ uint32_t*, uint32_t, uint16_t, uint8_t))          \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##src_data_type##_##dst_data_type##_uint32_t_uint64_t_Stub));                     \
                                                                                                                      \
        c_api_name(dst, src, src_offset, count);                                                                      \
        GlobalMockObject::verify();                                                                                   \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorCompute##class_name##src_data_type##dst_data_type, c_api_name##_sync_Succ)                       \
    {                                                                                                                 \
        __ubuf__ src_data_type* dst = reinterpret_cast<__ubuf__ src_data_type*>(11);                                  \
        __ubuf__ src_data_type* src = reinterpret_cast<__ubuf__ src_data_type*>(22);                                  \
        __ubuf__ uint32_t* src_offset = reinterpret_cast<__ubuf__ uint32_t*>(33);                                     \
        uint32_t count = static_cast<uint32_t>(44);                                                                   \
        MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                                         \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##src_data_type##_##dst_data_type##_set_vector_mask_Stub));                       \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type*, __ubuf__ uint32_t*, uint32_t, uint16_t, uint8_t))          \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##src_data_type##_##dst_data_type##_uint32_t_uint64_t_Stub));                     \
                                                                                                                      \
        c_api_name##_sync(dst, src, src_offset, count);                                                               \
        GlobalMockObject::verify();                                                                                   \
    }

// ==========asc_gather(int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, int16_t, uint16_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, uint16_t, uint16_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, half, uint16_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, bfloat16_t, uint16_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, int32_t, uint32_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, uint32_t, uint32_t);
TEST_VECTOR_COMPUTE_VGATHER_INSTR(Gather, asc_gather, vgather, float, uint32_t);
