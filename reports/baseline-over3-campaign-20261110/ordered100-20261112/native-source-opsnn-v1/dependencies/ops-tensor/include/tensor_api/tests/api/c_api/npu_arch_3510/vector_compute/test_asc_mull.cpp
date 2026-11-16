/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class TestVectorComputeMull : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void vmull_uint32_Stub(
    vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
{}
void vmull_int32_Stub(
    vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
{}
} // namespace

TEST_F(TestVectorComputeMull, c_api_mull_uint32_Succ)
{
    vector_uint32_t dst0;
    vector_uint32_t dst1;
    vector_uint32_t src0;
    vector_uint32_t src1;
    vector_bool mask;
    MOCKER_CPP(vmull, void(vector_uint32_t&, vector_uint32_t&, vector_uint32_t, vector_uint32_t, vector_bool))
        .times(1)
        .will(invoke(vmull_uint32_Stub));
    asc_mull(dst0, dst1, src0, src1, mask);
}

TEST_F(TestVectorComputeMull, c_api_mull_int32_Succ)
{
    vector_int32_t dst0;
    vector_int32_t dst1;
    vector_int32_t src0;
    vector_int32_t src1;
    vector_bool mask;
    MOCKER_CPP(vmull, void(vector_int32_t&, vector_int32_t&, vector_int32_t, vector_int32_t, vector_bool))
        .times(1)
        .will(invoke(vmull_int32_Stub));
    asc_mull(dst0, dst1, src0, src1, mask);
}
