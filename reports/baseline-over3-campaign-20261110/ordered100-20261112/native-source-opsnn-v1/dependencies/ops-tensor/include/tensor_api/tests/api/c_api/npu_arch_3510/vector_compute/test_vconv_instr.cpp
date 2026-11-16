/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_vcvt_instr_utils.h"

// ==========asc_e5m22float==========
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt, asc_e5m22float, vcvt, vector_float, vector_fp8_e5m2_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt_v2, asc_e5m22float_v2, vcvt, vector_float, vector_fp8_e5m2_t, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt_v3, asc_e5m22float_v3, vcvt, vector_float, vector_fp8_e5m2_t, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt_v4, asc_e5m22float_v4, vcvt, vector_float, vector_fp8_e5m2_t, 3);

// ==========asc_int322int64==========
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt, asc_int322int64, vcvt, vector_int64_t, vector_int32_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt_v2, asc_int322int64_v2, vcvt, vector_int64_t, vector_int32_t, 1);

// ==========asc_uint82uint16==========
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt, asc_uint82uint16, vcvt, vector_uint16_t, vector_uint8_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Vcvt_v2, asc_uint82uint16_v2, vcvt, vector_uint16_t, vector_uint8_t, 1);
