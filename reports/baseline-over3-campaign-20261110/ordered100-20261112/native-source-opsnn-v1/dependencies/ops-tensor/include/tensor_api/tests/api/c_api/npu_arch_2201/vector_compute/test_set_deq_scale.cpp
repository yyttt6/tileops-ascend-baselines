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
#include <cstring>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TestSetDeqScale : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {

void set_deqscale_ubuf_float_int16_t_bool_arr_Stub(uint64_t addr) {}

void set_deqscale_half_Stub(half scale_value)
{
    half val = 11;
    EXPECT_EQ(scale_value, val);
}

void set_deqscale_float_int16_t_bool_Stub(uint64_t config)
{
    int16_t offset_value = 1;
    bool sign_mode_value = true;
    float scale_value = 2.0;
}

} // namespace

TEST_F(TestSetDeqScale, set_deqscale_ubuf_float_int16_t_bool_arr_Succ)
{
    constexpr uint8_t ASC_VDEQ_SIZE = 16;
    std::vector<uint64_t> tmp_vec(ASC_VDEQ_SIZE);
    __ubuf__ uint64_t* tmp = tmp_vec.data();
    float scale_arr[ASC_VDEQ_SIZE] = {0};
    int16_t offset_arr[ASC_VDEQ_SIZE] = {0};
    bool sign_mode_arr[ASC_VDEQ_SIZE] = {true};
    MOCKER_CPP(set_deqscale, void(uint64_t)).times(1).will(invoke(set_deqscale_ubuf_float_int16_t_bool_arr_Stub));
    asc_set_deq_scale(tmp, scale_arr, offset_arr, sign_mode_arr);
    GlobalMockObject::verify();
}

TEST_F(TestSetDeqScale, set_deqscale_float_int16_t_bool_Succ)
{
    int16_t offset = 1;
    bool sign_mode = true;
    float scale = 2.0;

    MOCKER_CPP(set_deqscale, void(uint64_t)).times(1).will(invoke(set_deqscale_float_int16_t_bool_Stub));

    asc_set_deq_scale(scale, offset, sign_mode);
    GlobalMockObject::verify();
}

TEST_F(TestSetDeqScale, set_deqscale_half_Succ)
{
    half scale = 11;

    MOCKER_CPP(set_deqscale, void(half)).times(1).will(invoke(set_deqscale_half_Stub));

    asc_set_deq_scale(scale);
    GlobalMockObject::verify();
}
