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
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TEST_ASC_UNLOCK : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void rls_buf_stub(pipe_t pipe, uint8_t mutex_id, bool mode)
{
    EXPECT_EQ(pipe, pipe_t::PIPE_MTE3);
    EXPECT_EQ(mutex_id, static_cast<uint8_t>(11));
    EXPECT_EQ(mode, false);
}

void rls_buf_non_block_stub(pipe_t pipe, uint8_t mutex_id, bool mode)
{
    EXPECT_EQ(pipe, pipe_t::PIPE_MTE3);
    EXPECT_EQ(mutex_id, static_cast<uint8_t>(11));
    EXPECT_EQ(mode, true);
}
} // namespace

TEST_F(TEST_ASC_UNLOCK, asc_unlock_DefaultBlockSucc)
{
    MOCKER_CPP(rls_buf, void(pipe_t, uint8_t, bool)).times(1).will(invoke(rls_buf_stub));

    asc_unlock(pipe_t::PIPE_MTE3, static_cast<uint8_t>(11));
    GlobalMockObject::verify();
}

TEST_F(TEST_ASC_UNLOCK, asc_unlock_BlockSucc)
{
    MOCKER_CPP(rls_buf, void(pipe_t, uint8_t, bool)).times(1).will(invoke(rls_buf_stub));

    asc_unlock(pipe_t::PIPE_MTE3, static_cast<uint8_t>(11), ASC_LOCK_BLOCK);
    GlobalMockObject::verify();
}

TEST_F(TEST_ASC_UNLOCK, asc_unlock_NonBlockSucc)
{
    MOCKER_CPP(rls_buf, void(pipe_t, uint8_t, bool)).times(1).will(invoke(rls_buf_non_block_stub));

    asc_unlock(pipe_t::PIPE_MTE3, static_cast<uint8_t>(11), ASC_LOCK_NON_BLOCK);
    GlobalMockObject::verify();
}
