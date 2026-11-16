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

class TestDataCachePreload3510 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void data_cache_preload_Stub(__gm__ uint64_t* address, int64_t offset)
{
    EXPECT_EQ(reinterpret_cast<__gm__ uint64_t*>(1), address);
    EXPECT_EQ(static_cast<int64_t>(2), offset);
}
} // namespace

TEST_F(TestDataCachePreload3510, data_cache_preload_Stub)
{
    __gm__ uint64_t* address = reinterpret_cast<__gm__ uint64_t*>(1);
    int64_t offset = 2;
    MOCKER(dc_preload, void(__gm__ uint64_t*, int64_t)).times(1).will(invoke(data_cache_preload_Stub));

    asc_datacache_preload(address, offset);
    GlobalMockObject::verify();
}
