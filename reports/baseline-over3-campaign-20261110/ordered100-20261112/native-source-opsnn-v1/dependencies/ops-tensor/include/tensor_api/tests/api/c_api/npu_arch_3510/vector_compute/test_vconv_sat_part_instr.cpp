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

// ==========asc_float2bfloat16==========
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd, asc_float2bfloat16_rd, vcvt, vector_bfloat16_t, vector_float, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_sat, asc_float2bfloat16_rd_sat, vcvt, vector_bfloat16_t, vector_float, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_v2, asc_float2bfloat16_rd_v2, vcvt, vector_bfloat16_t, vector_float, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    Vcvt_rd_sat_v2, asc_float2bfloat16_rd_sat_v2, vcvt, vector_bfloat16_t, vector_float, 3);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn, asc_float2bfloat16_rn, vcvt, vector_bfloat16_t, vector_float, 4);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat, asc_float2bfloat16_rn_sat, vcvt, vector_bfloat16_t, vector_float, 5);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v2, asc_float2bfloat16_rn_v2, vcvt, vector_bfloat16_t, vector_float, 6);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    Vcvt_rn_sat_v2, asc_float2bfloat16_rn_sat_v2, vcvt, vector_bfloat16_t, vector_float, 7);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna, asc_float2bfloat16_rna, vcvt, vector_bfloat16_t, vector_float, 8);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_sat, asc_float2bfloat16_rna_sat, vcvt, vector_bfloat16_t, vector_float, 9);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_v2, asc_float2bfloat16_rna_v2, vcvt, vector_bfloat16_t, vector_float, 10);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    Vcvt_rna_sat_v2, asc_float2bfloat16_rna_sat_v2, vcvt, vector_bfloat16_t, vector_float, 11);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru, asc_float2bfloat16_ru, vcvt, vector_bfloat16_t, vector_float, 12);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_sat, asc_float2bfloat16_ru_sat, vcvt, vector_bfloat16_t, vector_float, 13);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_v2, asc_float2bfloat16_ru_v2, vcvt, vector_bfloat16_t, vector_float, 14);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    Vcvt_ru_sat_v2, asc_float2bfloat16_ru_sat_v2, vcvt, vector_bfloat16_t, vector_float, 15);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz, asc_float2bfloat16_rz, vcvt, vector_bfloat16_t, vector_float, 16);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_sat, asc_float2bfloat16_rz_sat, vcvt, vector_bfloat16_t, vector_float, 17);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_v2, asc_float2bfloat16_rz_v2, vcvt, vector_bfloat16_t, vector_float, 18);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    Vcvt_rz_sat_v2, asc_float2bfloat16_rz_sat_v2, vcvt, vector_bfloat16_t, vector_float, 19);
