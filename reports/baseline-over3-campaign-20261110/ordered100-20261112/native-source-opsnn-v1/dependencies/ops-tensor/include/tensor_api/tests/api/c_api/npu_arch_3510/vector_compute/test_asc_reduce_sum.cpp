/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_REDUCE_SUM_H
#define TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_REDUCE_SUM_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

// 宏定义：为每种 (dst_type, src_type) 组合创建单独的测试类和测试用例
// 通过 index 参数区分不同的类型组合，避免类名和测试名冲突

#define TEST_REDUCE_SUM_INSTR(class_name, c_api_name, cce_name, dst_type, src_type, index)                \
                                                                                                          \
    class TestVectorCompute##class_name##_##dst_type##_##src_type##_CApi_##index : public testing::Test { \
    protected:                                                                                            \
        void SetUp() {}                                                                                   \
        void TearDown() {}                                                                                \
    };                                                                                                    \
                                                                                                          \
    namespace {                                                                                           \
    void cce_name##_##dst_type##_##src_type##_Stub_##index(                                               \
        dst_type& dst, src_type src, vector_bool mask, Literal mode)                                      \
    {}                                                                                                    \
    }                                                                                                     \
                                                                                                          \
    TEST_F(                                                                                               \
        TestVectorCompute##class_name##_##dst_type##_##src_type##_CApi_##index,                           \
        c_api_name##_##dst_type##_##src_type##_Succ)                                                      \
    {                                                                                                     \
        dst_type dst;                                                                                     \
        src_type src;                                                                                     \
        vector_bool mask;                                                                                 \
                                                                                                          \
        MOCKER_CPP(cce_name, void(dst_type&, src_type, vector_bool, Literal))                             \
            .times(1)                                                                                     \
            .will(invoke(cce_name##_##dst_type##_##src_type##_Stub_##index));                             \
                                                                                                          \
        c_api_name(dst, src, mask);                                                                       \
        GlobalMockObject::verify();                                                                       \
    }

// ==========asc_reduce_sum 测试用例==========
// int16 -> int32
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_int32_t, vector_int16_t, 0);
// int32 -> int32
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_int32_t, vector_int32_t, 1);
// uint16 -> uint32
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_uint32_t, vector_uint16_t, 2);
// uint32 -> uint32
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_uint32_t, vector_uint32_t, 3);
// half -> half
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_half, vector_half, 4);
// float -> float
TEST_REDUCE_SUM_INSTR(Vcadd, asc_reduce_sum, vcadd, vector_float, vector_float, 5);

#endif
