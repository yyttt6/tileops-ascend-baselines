/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_vcvt_instr_utils.h"
//================asc_float2e5m2_rn================
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn, asc_float2e5m2_rn, vcvt, vector_f8e5m2, vector_float, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v2, asc_float2e5m2_rn_v2, vcvt, vector_f8e5m2, vector_float, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v3, asc_float2e5m2_rn_v3, vcvt, vector_f8e5m2, vector_float, 3);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v4, asc_float2e5m2_rn_v4, vcvt, vector_f8e5m2, vector_float, 4);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat, asc_float2e5m2_rn_sat, vcvt, vector_f8e5m2, vector_float, 5);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v2, asc_float2e5m2_rn_sat_v2, vcvt, vector_f8e5m2, vector_float, 6);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v3, asc_float2e5m2_rn_sat_v3, vcvt, vector_f8e5m2, vector_float, 7);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v4, asc_float2e5m2_rn_sat_v4, vcvt, vector_f8e5m2, vector_float, 8);
