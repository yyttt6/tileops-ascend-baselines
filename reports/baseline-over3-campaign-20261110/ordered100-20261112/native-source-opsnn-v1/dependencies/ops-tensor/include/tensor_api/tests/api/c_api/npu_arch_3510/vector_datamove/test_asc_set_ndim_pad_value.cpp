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

#define TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(data_type, type_value, u64_value)                                  \
                                                                                                                   \
    class TestVectorDatamoveSetNdimPadValue##data_type##CApi : public testing::Test {                              \
    protected:                                                                                                     \
        void SetUp() {}                                                                                            \
        void TearDown() {}                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
                                                                                                                   \
    void set_pad_val_nddma_##data_type##_Stub(uint64_t config) { EXPECT_EQ(u64_value, config); }                   \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestVectorDatamoveSetNdimPadValue##data_type##CApi, asc_set_ndim_pad_value_##data_type##_Succ)          \
    {                                                                                                              \
        data_type value = type_value;                                                                              \
                                                                                                                   \
        MOCKER_CPP(set_pad_val_nddma, void(uint64_t)).times(1).will(invoke(set_pad_val_nddma_##data_type##_Stub)); \
                                                                                                                   \
        asc_set_ndim_pad_value(value);                                                                             \
        GlobalMockObject::verify();                                                                                \
    }

//===========asc_set_ndim_pad_value(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float)===========
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(int8_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(uint8_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(int16_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(uint16_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(half, 1, 0x3c00);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(bfloat16_t, 1, 0x3f80);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(int32_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(uint32_t, 1, 0x1);
TEST_VECTOR_DATAMOVE_SET_NDIM_PAD_VALUE(float, 1, 0x3f800000);
