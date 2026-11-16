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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

class TestAtomicSetStoreAtomicConfigCApi : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestAtomicSetStoreAtomicConfigCApi, set_st_atomic_cfg_Succ)
{
    uint16_t type = 1; // int32_t
    uint16_t op = 2;   // add
    asc_set_store_atomic_config_v2(type, op);
    GlobalMockObject::verify();
}
