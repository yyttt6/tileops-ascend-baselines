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
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TestSetGM2UBPad : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void set_pad_val_outtoub_Stub(uint64_t config)
{
    uint64_t conf = 123;
    EXPECT_EQ(conf, config);
}
} // namespace

TEST_F(TestSetGM2UBPad, set_gm2ub_pad_Succ)
{
    MOCKER(set_pad_val_outtoub, void(uint64_t)).times(1).will(invoke(set_pad_val_outtoub_Stub));

    uint32_t config = 123;
    asc_set_gm2ub_pad(config);
    GlobalMockObject::verify();
}
