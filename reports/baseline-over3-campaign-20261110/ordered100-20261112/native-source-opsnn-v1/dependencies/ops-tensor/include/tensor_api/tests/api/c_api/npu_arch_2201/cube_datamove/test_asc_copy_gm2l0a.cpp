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

#define TEST_CUBE_DATAMOVE_GM2L0A(class_name, c_api_name, cce_name, data_type)                                  \
                                                                                                                \
    class TestCubeDatamove##class_name##c_api_name##cce_name##data_type : public testing::Test {                \
    protected:                                                                                                  \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                           \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                        \
    };                                                                                                          \
                                                                                                                \
    void mock##class_name##c_api_name##data_type##Stub(                                                         \
        __ca__ data_type* dst, __gm__ data_type* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,   \
        uint16_t dst_gap, uint8_t sid, addr_cal_mode_t addr_cal_mode)                                           \
    {                                                                                                           \
        EXPECT_EQ(dst, reinterpret_cast<__ca__ data_type*>(11));                                                \
        EXPECT_EQ(src, reinterpret_cast<__gm__ data_type*>(22));                                                \
        EXPECT_EQ(base_idx, static_cast<uint16_t>(33));                                                         \
        EXPECT_EQ(repeat, static_cast<uint8_t>(44));                                                            \
        EXPECT_EQ(src_stride, static_cast<uint16_t>(55));                                                       \
        EXPECT_EQ(dst_gap, static_cast<uint16_t>(66));                                                          \
        EXPECT_EQ(addr_cal_mode, addr_cal_mode_t::inc);                                                         \
    }                                                                                                           \
                                                                                                                \
    TEST_F(TestCubeDatamove##class_name##c_api_name##cce_name##data_type, c_api_name##data_type##_config_succ)  \
    {                                                                                                           \
        __ca__ data_type* dst = reinterpret_cast<data_type*>(11);                                               \
        __cbuf__ data_type* src = reinterpret_cast<data_type*>(22);                                             \
                                                                                                                \
        uint16_t base_idx = static_cast<uint16_t>(33);                                                          \
        uint8_t repeat = static_cast<uint8_t>(44);                                                              \
        uint16_t src_stride = static_cast<uint16_t>(55);                                                        \
        uint16_t dst_gap = static_cast<uint16_t>(66);                                                           \
                                                                                                                \
        MOCKER_CPP(                                                                                             \
            cce_name, void(                                                                                     \
                          __ca__ data_type*, __gm__ data_type*, uint16_t, uint8_t, uint16_t, uint16_t, uint8_t, \
                          addr_cal_mode_t))                                                                     \
            .times(1)                                                                                           \
            .will(invoke(mock##class_name##c_api_name##data_type##Stub));                                       \
                                                                                                                \
        c_api_name(dst, src, base_idx, repeat, src_stride, dst_gap);                                            \
        GlobalMockObject::verify();                                                                             \
    }

TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, bfloat16_t);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, half);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, float);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, int32_t);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, uint8_t);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, int8_t);
TEST_CUBE_DATAMOVE_GM2L0A(CopyGm2l0a, asc_copy_gm2l0a, load_gm_to_ca, uint32_t);
