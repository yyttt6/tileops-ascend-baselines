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

#define TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(class_name, c_api_name, cce_name)  \
                                                                                     \
    class TestVectorDatamove##class_name##CApi : public testing::Test {              \
    protected:                                                                       \
        void SetUp() {}                                                              \
        void TearDown() {}                                                           \
    };                                                                               \
                                                                                     \
    namespace {                                                                      \
                                                                                     \
    void cce_name##_Stub(uint64_t config) {}                                         \
    }                                                                                \
                                                                                     \
    TEST_F(TestVectorDatamove##class_name##CApi, c_api_name##_Succ)                  \
    {                                                                                \
        uint64_t dst_stride = 0;                                                     \
        uint64_t src_stride = 0;                                                     \
                                                                                     \
        MOCKER_CPP(cce_name, void(uint64_t)).times(1).will(invoke(cce_name##_Stub)); \
                                                                                     \
        c_api_name(dst_stride, src_stride);                                          \
        GlobalMockObject::verify();                                                  \
    }

TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(SetNdimLoop0Stride, asc_set_ndim_loop0_stride, set_loop0_stride_nddma);
TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(SetNdimLoop1Stride, asc_set_ndim_loop1_stride, set_loop1_stride_nddma);
TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(SetNdimLoop2Stride, asc_set_ndim_loop2_stride, set_loop2_stride_nddma);
TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(SetNdimLoop3Stride, asc_set_ndim_loop3_stride, set_loop3_stride_nddma);
TEST_VECTOR_DATAMOVE_SET_NDIM_LOOP_STRIDE(SetNdimLoop4Stride, asc_set_ndim_loop4_stride, set_loop4_stride_nddma);
