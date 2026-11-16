/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_unary_instr_utils.h"
#define TEST_VECTOR_COMPUTE_VCGADD_INSTR(class_name, c_api_name, cce_name, dst_type, src_type)                        \
                                                                                                                      \
    class TestVectorCompute##class_name##_##dst_type##_##src_type##_CApi : public testing::Test {                     \
    protected:                                                                                                        \
        void SetUp() {}                                                                                               \
        void TearDown() {}                                                                                            \
    };                                                                                                                \
                                                                                                                      \
    namespace {                                                                                                       \
    void cce_name##_##dst_type##_##src_type##_Stub(dst_type& dst, src_type src0, vector_bool mask, Literal mode) {}   \
    }                                                                                                                 \
                                                                                                                      \
    TEST_F(TestVectorCompute##class_name##_##dst_type##_##src_type##_CApi, cce_name##_##dst_type##_##src_type##_Succ) \
    {                                                                                                                 \
        dst_type dst;                                                                                                 \
        src_type src0;                                                                                                \
        vector_bool mask;                                                                                             \
                                                                                                                      \
        MOCKER_CPP(cce_name, void(dst_type&, src_type, vector_bool, Literal))                                         \
            .times(1)                                                                                                 \
            .will(invoke(cce_name##_##dst_type##_##src_type##_Stub));                                                 \
                                                                                                                      \
        c_api_name(dst, src0, mask);                                                                                  \
        GlobalMockObject::verify();                                                                                   \
    }

TEST_VECTOR_COMPUTE_UNARY_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_half);
TEST_VECTOR_COMPUTE_UNARY_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_float);
TEST_VECTOR_COMPUTE_UNARY_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_uint32_t);
TEST_VECTOR_COMPUTE_UNARY_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_int32_t);
TEST_VECTOR_COMPUTE_VCGADD_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_uint32_t, vector_uint16_t);
TEST_VECTOR_COMPUTE_VCGADD_INSTR(Vcgadd, asc_reduce_sum_datablock, vcgadd, vector_int32_t, vector_int16_t);
