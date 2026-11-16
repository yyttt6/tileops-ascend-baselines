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

#define TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(class_name, c_api_name, cce_name, data_type, cce_type) \
                                                                                                    \
    class TestVectorCompute##class_name####data_type##CApi : public testing::Test {                 \
    protected:                                                                                      \
        void SetUp() {}                                                                             \
        void TearDown() {}                                                                          \
    };                                                                                              \
                                                                                                    \
    namespace {                                                                                     \
    void cce_name##_##data_type##_Stub(cce_type dst, __gm__ cce_type* src0, int16_t offset) {}      \
    }                                                                                               \
                                                                                                    \
    TEST_F(TestVectorCompute##class_name####data_type##CApi, c_api_name##_##data_type##_Succ)       \
    {                                                                                               \
        __gm__ data_type* dst;                                                                      \
        data_type src0;                                                                             \
        int16_t offset;                                                                             \
                                                                                                    \
        MOCKER_CPP(cce_name, void(cce_type, __gm__ cce_type*, int16_t))                             \
            .times(1)                                                                               \
            .will(invoke(cce_name##_##data_type##_Stub));                                           \
                                                                                                    \
        c_api_name(dst, src0);                                                                      \
        GlobalMockObject::verify();                                                                 \
    }

TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, uint8_t, uint8_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, uint16_t, uint16_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, uint32_t, uint32_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, uint64_t, uint64_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, int8_t, uint8_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, int16_t, uint16_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, int32_t, uint32_t);
TEST_VECTOR_COMPUTE_SCALAR_DEV_INSTR(WRITEGMDCACHE, asc_store_dev, st_dev, int64_t, uint64_t);
