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

class TestSimdAtomicGetStoreAtomicConfigCApi : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestSimdAtomicGetStoreAtomicConfigCApi, get_st_atomic_cfg_Succ)
{
    asc_store_atomic_config config;
    asc_get_store_atomic_config(config);
    uint16_t atomic_type = config.atomic_type;
    uint16_t atomic_op = config.atomic_op;
    EXPECT_EQ(atomic_type, 0);
    EXPECT_EQ(atomic_op, 0);
    asc_set_store_atomic_config_v1(atomic_type, atomic_op);
    GlobalMockObject::verify();
}
