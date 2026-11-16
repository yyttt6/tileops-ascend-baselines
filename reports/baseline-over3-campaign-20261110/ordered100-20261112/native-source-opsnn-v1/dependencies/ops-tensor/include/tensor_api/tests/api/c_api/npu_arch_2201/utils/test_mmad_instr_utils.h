/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_MMAD_INSTR_UTILS_H
#define TESTS_UNIT_BASIC_API_UT_TESTCASE_C_API_NPU_ARCH_2201_UTILS_TEST_MMAD_INSTR_UTILS_H

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_CUBE_COMPUTE_MMAD_INSTR(class_name, capi_name, cce_name, dst_data_type, src_data_type)                    \
                                                                                                                       \
    class Test##class_name##CAPi : public testing::Test {                                                              \
    protected:                                                                                                         \
        void SetUp() { g_coreType = C_API_AIC_TYPE; }                                                                  \
        void TearDown() { g_coreType = C_API_AIV_TYPE; }                                                               \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void mad_no_mask_Stub(                                                                                             \
        __cc__ dst_data_type* c, __ca__ src_data_type* a, __cb__ src_data_type* b, uint16_t m, uint16_t k, uint16_t n, \
        uint8_t unitFlag, bool k_direction_align, bool cmatrixSource, bool cmatrixInitVal)                             \
    {                                                                                                                  \
        EXPECT_EQ(c, reinterpret_cast<__cc__ dst_data_type*>(1));                                                      \
        EXPECT_EQ(a, reinterpret_cast<__ca__ src_data_type*>(2));                                                      \
        EXPECT_EQ(b, reinterpret_cast<__cb__ src_data_type*>(3));                                                      \
        EXPECT_EQ(m, static_cast<uint16_t>(4));                                                                        \
        EXPECT_EQ(k, static_cast<uint16_t>(5));                                                                        \
        EXPECT_EQ(n, static_cast<uint16_t>(6));                                                                        \
        EXPECT_EQ(unitFlag, static_cast<uint8_t>(1));                                                                  \
        EXPECT_EQ(cmatrixSource, false);                                                                               \
        EXPECT_EQ(cmatrixInitVal, true);                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    void mad_have_mask_Stub(                                                                                           \
        __cc__ dst_data_type* c_matrix, __ca__ src_data_type* a_matrix, __cb__ src_data_type* b_matrix,                \
        uint16_t left_height, uint16_t right_width, uint16_t n_dim, uint8_t feat_offset, uint8_t smask_offset,         \
        uint8_t unit_flag, bool k_direction_align, bool is_weight_offset, bool c_matrix_source,                        \
        bool c_matrix_init_val)                                                                                        \
    {                                                                                                                  \
        EXPECT_EQ(c_matrix, reinterpret_cast<__cc__ dst_data_type*>(1));                                               \
        EXPECT_EQ(a_matrix, reinterpret_cast<__ca__ src_data_type*>(2));                                               \
        EXPECT_EQ(b_matrix, reinterpret_cast<__cb__ src_data_type*>(3));                                               \
        EXPECT_EQ(left_height, static_cast<uint16_t>(4));                                                              \
        EXPECT_EQ(right_width, static_cast<uint16_t>(5));                                                              \
        EXPECT_EQ(n_dim, static_cast<uint16_t>(6));                                                                    \
        EXPECT_EQ(feat_offset, static_cast<uint8_t>(7));                                                               \
        EXPECT_EQ(unit_flag, static_cast<uint8_t>(1));                                                                 \
        EXPECT_EQ(k_direction_align, true);                                                                            \
        EXPECT_EQ(is_weight_offset, true);                                                                             \
        EXPECT_EQ(c_matrix_source, false);                                                                             \
        EXPECT_EQ(c_matrix_init_val, true);                                                                            \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(Test##class_name##CAPi, mmad_Succ)                                                                          \
    {                                                                                                                  \
        __cc__ dst_data_type* c = reinterpret_cast<__cc__ dst_data_type*>(1);                                          \
        __ca__ src_data_type* a = reinterpret_cast<__ca__ src_data_type*>(2);                                          \
        __cb__ src_data_type* b = reinterpret_cast<__cb__ src_data_type*>(3);                                          \
        uint16_t m = 4;                                                                                                \
        uint16_t k = 5;                                                                                                \
        uint16_t n = 6;                                                                                                \
        uint8_t unitFlag = 1;                                                                                          \
        bool k_direction_align = true;                                                                                 \
        bool cmatrixSource = false;                                                                                    \
        bool cmatrixInitVal = true;                                                                                    \
        MOCKER(                                                                                                        \
            cce_name, void(                                                                                            \
                          __cc__ dst_data_type*, __ca__ src_data_type*, __cb__ src_data_type*, uint16_t, uint16_t,     \
                          uint16_t, uint8_t, bool, bool, bool))                                                        \
            .times(1)                                                                                                  \
            .will(invoke(mad_no_mask_Stub));                                                                           \
                                                                                                                       \
        capi_name(c, a, b, m, k, n, unitFlag, k_direction_align, cmatrixSource, cmatrixInitVal);                       \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(Test##class_name##CAPi, mmad_sync_Succ)                                                                     \
    {                                                                                                                  \
        __cc__ dst_data_type* c = reinterpret_cast<__cc__ dst_data_type*>(1);                                          \
        __ca__ src_data_type* a = reinterpret_cast<__ca__ src_data_type*>(2);                                          \
        __cb__ src_data_type* b = reinterpret_cast<__cb__ src_data_type*>(3);                                          \
        uint16_t m = 4;                                                                                                \
        uint16_t k = 5;                                                                                                \
        uint16_t n = 6;                                                                                                \
        uint8_t unitFlag = 1;                                                                                          \
        bool k_direction_align = true;                                                                                 \
        bool cmatrixSource = false;                                                                                    \
        bool cmatrixInitVal = true;                                                                                    \
        MOCKER(                                                                                                        \
            cce_name, void(                                                                                            \
                          __cc__ dst_data_type*, __ca__ src_data_type*, __cb__ src_data_type*, uint16_t, uint16_t,     \
                          uint16_t, uint8_t, bool, bool, bool))                                                        \
            .times(1)                                                                                                  \
            .will(invoke(mad_no_mask_Stub));                                                                           \
                                                                                                                       \
        capi_name(c, a, b, m, k, n, unitFlag, k_direction_align, cmatrixSource, cmatrixInitVal);                       \
        GlobalMockObject::verify();                                                                                    \
    }

#endif
