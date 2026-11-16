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

#define TEST_SCALAR_COMPUTE_ASC_FLOAT2INT32_INSTR(class_name, c_api_name, cce_name, data_type) \
                                                                                               \
    class TestScalarCompute##class_name####data_type##CApi : public testing::Test {            \
    protected:                                                                                 \
        void SetUp() {}                                                                        \
        void TearDown() {}                                                                     \
    };                                                                                         \
                                                                                               \
    namespace {                                                                                \
    int64_t cce_name##_Stub(float value) { return 0; }                                         \
    }                                                                                          \
                                                                                               \
    TEST_F(TestScalarCompute##class_name####data_type##CApi, c_api_name##_Succ)                \
    {                                                                                          \
        int32_t dst;                                                                           \
        float src;                                                                             \
                                                                                               \
        MOCKER_CPP(cce_name, int64_t(float)).times(1).will(invoke(cce_name##_Stub));           \
                                                                                               \
        dst = c_api_name(src);                                                                 \
        GlobalMockObject::verify();                                                            \
    }

TEST_SCALAR_COMPUTE_ASC_FLOAT2INT32_INSTR(Float2Int32, asc_float2int32_rd, conv_f322s32f, Rd);
TEST_SCALAR_COMPUTE_ASC_FLOAT2INT32_INSTR(Float2Int32, asc_float2int32_ru, conv_f322s32c, Ru);
TEST_SCALAR_COMPUTE_ASC_FLOAT2INT32_INSTR(Float2Int32, asc_float2int32_rn, conv_f322s32r, Rn);
TEST_SCALAR_COMPUTE_ASC_FLOAT2INT32_INSTR(Float2Int32, asc_float2int32_rna, conv_f322s32a, Rna);
