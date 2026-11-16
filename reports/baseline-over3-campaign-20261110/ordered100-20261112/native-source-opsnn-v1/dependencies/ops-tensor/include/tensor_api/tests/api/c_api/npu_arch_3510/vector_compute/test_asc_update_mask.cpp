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

class TestVectorComputeUpdateMask : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
vector_bool plt_b8_Stub(uint32_t& scalar, Literal mode) { return vector_bool{}; }

vector_bool plt_b16_Stub(uint32_t& scalar, Literal mode) { return vector_bool{}; }

vector_bool plt_b32_Stub(uint32_t& scalar, Literal mode) { return vector_bool{}; }
} // namespace

TEST_F(TestVectorComputeUpdateMask, c_api_update_mask_b8_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b8, vector_bool(uint32_t&, Literal)).times(1).will(invoke(plt_b8_Stub));
    asc_update_mask_b8(scalar);
}

TEST_F(TestVectorComputeUpdateMask, c_api_update_mask_b16_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b16, vector_bool(uint32_t&, Literal)).times(1).will(invoke(plt_b16_Stub));
    asc_update_mask_b16(scalar);
}

TEST_F(TestVectorComputeUpdateMask, c_api_update_mask_b32_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b32, vector_bool(uint32_t&, Literal)).times(1).will(invoke(plt_b32_Stub));
    asc_update_mask_b32(scalar);
}

class TestVectorComputeUpdateMaskCube : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

TEST_F(TestVectorComputeUpdateMaskCube, c_api_update_mask_b8_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b8, vector_bool(uint32_t&, Literal)).times(0).will(invoke(plt_b8_Stub));
    asc_update_mask_b8(scalar);
}

TEST_F(TestVectorComputeUpdateMaskCube, c_api_update_mask_b16_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b16, vector_bool(uint32_t&, Literal)).times(0).will(invoke(plt_b16_Stub));
    asc_update_mask_b16(scalar);
}

TEST_F(TestVectorComputeUpdateMaskCube, c_api_update_mask_b32_Succ)
{
    uint32_t scalar;
    MOCKER_CPP(plt_b32, vector_bool(uint32_t&, Literal)).times(0).will(invoke(plt_b32_Stub));
    asc_update_mask_b32(scalar);
}
