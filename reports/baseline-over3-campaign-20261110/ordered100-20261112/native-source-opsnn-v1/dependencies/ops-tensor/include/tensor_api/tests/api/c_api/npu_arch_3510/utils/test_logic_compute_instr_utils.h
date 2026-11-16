/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_LOGIC_COMPUTE_INSTR_UTILS_H
#define TESTS_API_C_API_NPU_ARCH_3510_UTILS_TEST_LOGIC_COMPUTE_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_LOGIC_COMPUTE_INSTR(class_name, c_api_name, cce_name, data_type)                                          \
                                                                                                                       \
    class TestLogicCompute##class_name####data_type##CApi : public testing::Test {                                     \
    protected:                                                                                                         \
        void SetUp() {}                                                                                                \
        void TearDown() {}                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void cce_name##_##data_type##_Stub(data_type& dst, data_type src0, data_type src1, vector_bool mask, Literal mode) \
    {}                                                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestLogicCompute##class_name####data_type##CApi, c_api_name##_##data_type##_Succ)                           \
    {                                                                                                                  \
        data_type dst;                                                                                                 \
        data_type src0;                                                                                                \
        data_type src1;                                                                                                \
        vector_bool mask;                                                                                              \
                                                                                                                       \
        MOCKER_CPP(cce_name, void(data_type&, data_type, data_type, vector_bool, Literal))                             \
            .times(1)                                                                                                  \
            .will(invoke(cce_name##_##data_type##_Stub));                                                              \
                                                                                                                       \
        c_api_name(dst, src0, src1, mask);                                                                             \
        GlobalMockObject::verify();                                                                                    \
    }

#endif
