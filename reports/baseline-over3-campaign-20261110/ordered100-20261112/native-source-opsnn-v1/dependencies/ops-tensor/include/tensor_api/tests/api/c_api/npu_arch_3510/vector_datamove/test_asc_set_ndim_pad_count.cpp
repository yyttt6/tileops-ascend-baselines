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

class TestSetNdimPadCountCApi : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void set_pad_cnt_nddma_stub(uint64_t config) { EXPECT_EQ(0x0807060504030201, config); }
} // namespace

TEST_F(TestSetNdimPadCountCApi, asc_set_ndim_pad_count_Succ)
{
    asc_ndim_pad_count_config config;
    config.loop1_lp_count = 1;
    config.loop1_rp_count = 2;
    config.loop2_lp_count = 3;
    config.loop2_rp_count = 4;
    config.loop3_lp_count = 5;
    config.loop3_rp_count = 6;
    config.loop4_lp_count = 7;
    config.loop4_rp_count = 8;
    MOCKER_CPP(set_pad_cnt_nddma, void(uint64_t)).times(1).will(invoke(set_pad_cnt_nddma_stub));
    asc_set_ndim_pad_count(config);
    GlobalMockObject::verify();
}
