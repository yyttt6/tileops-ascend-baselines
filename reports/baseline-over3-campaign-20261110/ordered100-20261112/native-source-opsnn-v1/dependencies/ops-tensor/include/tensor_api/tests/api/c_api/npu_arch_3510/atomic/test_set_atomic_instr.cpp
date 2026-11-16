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

#define TEST_ATOMIC_SET_ATOMIC_INSTR(class_name, c_api_name) \
                                                             \
    class TestAtomic##class_name : public testing::Test {    \
    protected:                                               \
        void SetUp() {}                                      \
        void TearDown() {}                                   \
    };                                                       \
                                                             \
    TEST_F(TestAtomic##class_name, c_api_name##_Succ)        \
    {                                                        \
        c_api_name();                                        \
        GlobalMockObject::verify();                          \
    }

TEST_ATOMIC_SET_ATOMIC_INSTR(SetAtomicAddFloatCApi, asc_set_atomic_add_float);
TEST_ATOMIC_SET_ATOMIC_INSTR(SetAtomicMaxFloat16CApi, asc_set_atomic_max_float16);
