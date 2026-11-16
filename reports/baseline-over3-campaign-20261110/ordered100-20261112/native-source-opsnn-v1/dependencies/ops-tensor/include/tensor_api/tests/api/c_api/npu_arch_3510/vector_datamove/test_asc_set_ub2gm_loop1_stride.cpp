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

class TestSetUb2GmLp1Stride : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void set_loop1_stride_ubtoout_stub(uint64_t config) {}
} // namespace

TEST_F(TestSetUb2GmLp1Stride, c_api_asc_set_ub2gm_loop1_stride_Succ)
{
    uint64_t loop1_src_stride = 1;
    uint64_t loop1_dst_stride = 1;
    MOCKER_CPP(set_loop1_stride_ubtoout, void(uint64_t)).times(1).will(invoke(set_loop1_stride_ubtoout_stub));
    asc_set_ub2gm_loop1_stride(loop1_src_stride, loop1_dst_stride);
    GlobalMockObject::verify();
}
