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

class TestVectorComputeFrequencyHistogram : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void dhistv2_Stub(vector_u16& dst, vector_u8 src, vector_bool mask, uint32_t bin) {}
} // namespace

TEST_F(TestVectorComputeFrequencyHistogram, c_api_frequency_histogram_Succ)
{
    vector_u16 dst;
    vector_u8 src;
    vector_bool mask;
    MOCKER_CPP(dhistv2, void(vector_u16&, vector_u8, vector_bool, uint32_t)).times(2).will(invoke(dhistv2_Stub));
    asc_frequency_histogram_bin0(dst, src, mask);
    asc_frequency_histogram_bin1(dst, src, mask);
}
