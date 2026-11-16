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

#define TEST_VECTOR_COMPUTE_CREATE_MASK(class_name, c_api_name, cce_name)                  \
                                                                                           \
    class TestVectorCompute##class_name##_CApi : public testing::Test {                    \
    protected:                                                                             \
        void SetUp() {}                                                                    \
        void TearDown() {}                                                                 \
    };                                                                                     \
                                                                                           \
    namespace {                                                                            \
    vector_bool cce_name##_Stub(Literal mode) { return vector_bool{}; }                    \
    }                                                                                      \
                                                                                           \
    TEST_F(TestVectorCompute##class_name##_CApi, c_api_name##_Succ)                        \
    {                                                                                      \
        MOCKER_CPP(cce_name, vector_bool(Literal)).times(1).will(invoke(cce_name##_Stub)); \
                                                                                           \
        c_api_name(PAT_VL16);                                                              \
        GlobalMockObject::verify();                                                        \
    }

TEST_VECTOR_COMPUTE_CREATE_MASK(Pset_b8, asc_create_mask_b8, pset_b8);
TEST_VECTOR_COMPUTE_CREATE_MASK(Pset_b16, asc_create_mask_b16, pset_b16);
TEST_VECTOR_COMPUTE_CREATE_MASK(Pset_b32, asc_create_mask_b32, pset_b32);
