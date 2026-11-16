/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_FIXPIPE_L0C2L1_INSTR_UTILS_H
#define TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_FIXPIPE_L0C2L1_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_CUBE_COMPUTE_FIXPIPE_INSTR(class_name, c_api_name, cce_name, dst_data_type, src_data_type)        \
                                                                                                               \
    class TestCubeDatamove##class_name##_##dst_data_type##src_data_type : public testing::Test {               \
    protected:                                                                                                 \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                          \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                       \
    };                                                                                                         \
                                                                                                               \
    namespace {                                                                                                \
    void c_api_name##_##dst_data_type##_##src_data_type##_Stub(                                                \
        __gm__ dst_data_type* dst_in, __cc__ src_data_type* src_in, uint8_t sid_in, uint16_t n_size_in,        \
        uint16_t m_size_in, uint32_t dst_stride_dst_d_in, uint16_t src_stride_in, uint8_t unit_flag_mode_in,   \
        QuantMode_t quant_pre_in, uint8_t relu_pre_in, bool channel_split_in, bool nz2nd_en_in)                \
    {                                                                                                          \
        __gm__ dst_data_type* dst = reinterpret_cast<__gm__ dst_data_type*>(1);                                \
        __cc__ src_data_type* src = reinterpret_cast<__gm__ src_data_type*>(2);                                \
        uint8_t sid = 3;                                                                                       \
        uint16_t n_size = 4;                                                                                   \
        uint16_t m_size = 5;                                                                                   \
        uint32_t dst_stride_dst_d = 6;                                                                         \
        uint16_t src_stride = 7;                                                                               \
        uint8_t unit_flag_mode = 8;                                                                            \
        uint8_t relu_pre = 10;                                                                                 \
        bool channel_split = true;                                                                             \
        bool nz2nd_en = true;                                                                                  \
                                                                                                               \
        EXPECT_EQ(dst, dst_in);                                                                                \
        EXPECT_EQ(src, src_in);                                                                                \
        EXPECT_EQ(n_size, n_size_in);                                                                          \
        EXPECT_EQ(m_size, m_size_in);                                                                          \
        EXPECT_EQ(dst_stride_dst_d, dst_stride_dst_d_in);                                                      \
        EXPECT_EQ(src_stride, src_stride_in);                                                                  \
        EXPECT_EQ(unit_flag_mode, unit_flag_mode_in);                                                          \
        EXPECT_EQ(relu_pre, relu_pre_in);                                                                      \
        EXPECT_EQ(channel_split, channel_split_in);                                                            \
        EXPECT_EQ(nz2nd_en, nz2nd_en_in);                                                                      \
    }                                                                                                          \
    }                                                                                                          \
                                                                                                               \
    TEST_F(                                                                                                    \
        TestCubeDatamove##class_name##_##dst_data_type##src_data_type,                                         \
        c_api_name##_##dst_data_type##_##src_data_type##_Succ)                                                 \
    {                                                                                                          \
        __gm__ dst_data_type* dst = reinterpret_cast<__gm__ dst_data_type*>(1);                                \
        __cc__ src_data_type* src = reinterpret_cast<__cc__ src_data_type*>(2);                                \
        uint8_t sid = 3;                                                                                       \
        uint16_t n_size = 4;                                                                                   \
        uint16_t m_size = 5;                                                                                   \
        uint32_t dst_stride_dst_d = 6;                                                                         \
        uint16_t src_stride = 7;                                                                               \
        uint8_t unit_flag_mode = 8;                                                                            \
        uint64_t quant_pre = NoQuant;                                                                          \
        uint8_t relu_pre = 10;                                                                                 \
        bool channel_split = true;                                                                             \
        bool nz2nd_en = true;                                                                                  \
                                                                                                               \
        MOCKER_CPP(                                                                                            \
            cce_name, void(                                                                                    \
                          __gm__ dst_data_type*, __cc__ src_data_type*, uint8_t, uint16_t, uint16_t, uint32_t, \
                          uint16_t, uint8_t, QuantMode_t, uint8_t, bool, bool))                                \
            .times(1)                                                                                          \
            .will(invoke(c_api_name##_##dst_data_type##_##src_data_type##_Stub));                              \
                                                                                                               \
        c_api_name(                                                                                            \
            dst, src, n_size, m_size, dst_stride_dst_d, src_stride, unit_flag_mode, quant_pre, relu_pre,       \
            channel_split, nz2nd_en);                                                                          \
        GlobalMockObject::verify();                                                                            \
    }

#endif
