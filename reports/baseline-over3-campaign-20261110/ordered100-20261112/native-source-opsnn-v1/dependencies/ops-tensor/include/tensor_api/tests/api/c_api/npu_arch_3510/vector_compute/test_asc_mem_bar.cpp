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

#define TEST_MEM_BAR_INSTR(class_name, c_api_name, cce_name, mem_type)                           \
                                                                                                 \
    class Test##class_name##mem_type##CApi : public testing::Test {                              \
    protected:                                                                                   \
        void SetUp() {}                                                                          \
        void TearDown() {}                                                                       \
    };                                                                                           \
                                                                                                 \
    namespace {                                                                                  \
    void cce_name##_##mem_type##_Stub(Literal dist) {}                                           \
    }                                                                                            \
                                                                                                 \
    TEST_F(Test##class_name##mem_type##CApi, mem_type##_Succ)                                    \
    {                                                                                            \
        MOCKER_CPP(cce_name, void(Literal)).times(1).will(invoke(cce_name##_##mem_type##_Stub)); \
                                                                                                 \
        c_api_name(mem_type);                                                                    \
        GlobalMockObject::verify();                                                              \
    }

TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VV_ALL)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VST_VLD)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VLD_VST)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VST_VST)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VS_ALL)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VST_LD)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VLD_ST)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, VST_ST)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, SV_ALL)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, ST_VLD)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, LD_VST)
TEST_MEM_BAR_INSTR(MemBar, asc_mem_bar, mem_bar, ST_VST)
