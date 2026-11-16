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

#define TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(class_name, c_api_name, cce_name, data_type, val_type)                    \
                                                                                                                   \
    class TestCubeDmamove##class_name##data_type##val_type : public testing::Test {                                \
    protected:                                                                                                     \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                              \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                           \
    };                                                                                                             \
                                                                                                                   \
    namespace {                                                                                                    \
                                                                                                                   \
    void cce_name##_##data_type##_int64_t_##val_type##_Stub(__ca__ data_type* dst, int64_t repeat, val_type value) \
    {                                                                                                              \
        EXPECT_EQ(dst, reinterpret_cast<__ca__ data_type*>(11));                                                   \
        EXPECT_EQ(repeat, static_cast<int64_t>(0));                                                                \
        EXPECT_EQ(value, static_cast<val_type>(1));                                                                \
    }                                                                                                              \
                                                                                                                   \
    void cce_name##_##data_type##_##val_type##_InitFillL0AValueConfig(asc_fill_value_config& config)               \
    {                                                                                                              \
        config.repeat = static_cast<uint64_t>(0);                                                                  \
        config.blk_num = static_cast<uint64_t>(0);                                                                 \
        config.dst_gap = static_cast<uint64_t>(0);                                                                 \
    }                                                                                                              \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestCubeDmamove##class_name##data_type##val_type, c_api_name##_FillValueConfig_Succ)                    \
    {                                                                                                              \
        __ca__ data_type* dst = reinterpret_cast<__ca__ data_type*>(11);                                           \
        val_type value = 1;                                                                                        \
                                                                                                                   \
        asc_fill_value_config config;                                                                              \
        cce_name##_##data_type##_##val_type##_InitFillL0AValueConfig(config);                                      \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(__ca__ data_type*, int64_t, val_type))                                           \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_int64_t_##val_type##_Stub));                                     \
                                                                                                                   \
        c_api_name(dst, value, config);                                                                            \
        GlobalMockObject::verify();                                                                                \
    }                                                                                                              \
                                                                                                                   \
    TEST_F(TestCubeDmamove##class_name##data_type##val_type, c_api_name##_sync_Succ)                               \
    {                                                                                                              \
        __ca__ data_type* dst = reinterpret_cast<__ca__ data_type*>(11);                                           \
        val_type value = 1;                                                                                        \
                                                                                                                   \
        asc_fill_value_config config;                                                                              \
        cce_name##_##data_type##_##val_type##_InitFillL0AValueConfig(config);                                      \
                                                                                                                   \
        MOCKER_CPP(cce_name, void(__ca__ data_type*, int64_t, val_type))                                           \
            .times(1)                                                                                              \
            .will(invoke(cce_name##_##data_type##_int64_t_##val_type##_Stub));                                     \
                                                                                                                   \
        c_api_name##_sync(dst, value, config);                                                                     \
        GlobalMockObject::verify();                                                                                \
    }

// ==========asc_fill_l0a(half/float/int16_t/int32_t/uint16_t/uint32_t/bfloat16_t)==========
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, half, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, half, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, float, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, float, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, int16_t, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, int16_t, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, int32_t, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, int32_t, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, uint16_t, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, uint16_t, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, uint32_t, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix, uint32_t, uint32_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix_bf16, bfloat16_t, bfloat16_t);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix_h, bfloat16_t, half);
TEST_CUBE_DMAMOVE_FILL_L0A_VALUE(FillL0AValue, asc_fill_l0a, create_ca_matrix_ui, bfloat16_t, uint32_t);
