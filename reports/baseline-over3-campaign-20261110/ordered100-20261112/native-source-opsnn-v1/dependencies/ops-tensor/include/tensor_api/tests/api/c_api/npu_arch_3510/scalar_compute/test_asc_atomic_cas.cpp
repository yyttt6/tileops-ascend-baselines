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

using namespace __asc_aicore;

#define TEST_ATOMIC_CAS_GM(data_type)                                                                           \
    class TestAtomicCas##Gm##_##data_type##CApi : public testing::Test {                                        \
    protected:                                                                                                  \
        void SetUp() {}                                                                                         \
        void TearDown() {}                                                                                      \
    };                                                                                                          \
                                                                                                                \
    namespace {                                                                                                 \
    data_type atomicCAS##_##gm##_##data_type##Stub(__gm__ data_type* address, data_type compare, data_type val) \
    {                                                                                                           \
        return compare;                                                                                         \
    }                                                                                                           \
    }                                                                                                           \
                                                                                                                \
    TEST_F(TestAtomicCas##Gm##_##data_type##CApi, c_api_atomic_cas_gm_##data_type##_Succ)                       \
    {                                                                                                           \
        data_type addr = 10;                                                                                    \
        data_type compare = 10;                                                                                 \
        data_type val = 5;                                                                                      \
        MOCKER_CPP(atomicCAS, data_type(__gm__ data_type*, data_type, data_type))                               \
            .times(1)                                                                                           \
            .will(invoke(atomicCAS##_##gm##_##data_type##Stub));                                                \
        data_type res = asc_atomic_cas((__gm__ data_type*)&addr, compare, val);                                 \
        EXPECT_EQ(res, compare);                                                                                \
        GlobalMockObject::verify();                                                                             \
    }

TEST_ATOMIC_CAS_GM(int32_t)
TEST_ATOMIC_CAS_GM(uint32_t)
TEST_ATOMIC_CAS_GM(float)
TEST_ATOMIC_CAS_GM(int64_t)
TEST_ATOMIC_CAS_GM(uint64_t)
