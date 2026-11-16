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

__aicore__ inline void copy_cbuf_to_gm_stub(
    __gm__ void* dst, __cbuf__ void* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    EXPECT_EQ(dst, reinterpret_cast<__gm__ void*>(11));
    EXPECT_EQ(src, reinterpret_cast<__cbuf__ void*>(22));
    EXPECT_EQ(sid, static_cast<uint8_t>(0));
    EXPECT_EQ(nBurst, static_cast<uint16_t>(33));
    EXPECT_EQ(lenBurst, static_cast<uint16_t>(44));
    EXPECT_EQ(srcGap, static_cast<uint16_t>(55));
    EXPECT_EQ(dstGap, static_cast<uint16_t>(66));
}

class TEST_COPY_L1_TO_GM : public testing::Test {
protected:
    void SetUp() { g_coreType = C_API_AIC_TYPE; }
    void TearDown() { g_coreType = C_API_AIV_TYPE; }
};

TEST_F(TEST_COPY_L1_TO_GM, TEST_COPY_L1_TO_GM)
{
    MOCKER_CPP(copy_cbuf_to_gm, void(__gm__ void*, __cbuf__ void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))
        .times(1)
        .will(invoke(copy_cbuf_to_gm_stub));

    __gm__ void* dst = reinterpret_cast<__gm__ void*>(11);
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(22);

    uint16_t nBurst = static_cast<uint16_t>(33);
    uint16_t lenBurst = static_cast<uint16_t>(44);
    uint16_t srcGap = static_cast<uint16_t>(55);
    uint16_t dstGap = static_cast<uint16_t>(66);

    asc_copy_l12gm(dst, src, nBurst, lenBurst, srcGap, dstGap);
    GlobalMockObject::verify();
}

TEST_F(TEST_COPY_L1_TO_GM, TEST_COPY_L1_TO_GM_SYNC)
{
    MOCKER_CPP(copy_cbuf_to_gm, void(__gm__ void*, __cbuf__ void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))
        .times(1)
        .will(invoke(copy_cbuf_to_gm_stub));

    __gm__ void* dst = reinterpret_cast<__gm__ void*>(11);
    __cbuf__ void* src = reinterpret_cast<__cbuf__ void*>(22);

    uint16_t nBurst = static_cast<uint16_t>(33);
    uint16_t lenBurst = static_cast<uint16_t>(44);
    uint16_t srcGap = static_cast<uint16_t>(55);
    uint16_t dstGap = static_cast<uint16_t>(66);

    asc_copy_l12gm_sync(dst, src, nBurst, lenBurst, srcGap, dstGap);
    GlobalMockObject::verify();
}
