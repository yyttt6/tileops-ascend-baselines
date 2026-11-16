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

TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rn, asc_bfloat162e1m2x2_rn, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rn_v1, asc_bfloat162e1m2x2_rn_v2, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rn_v2, asc_bfloat162e1m2x2_rn_v3, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rn_v3, asc_bfloat162e1m2x2_rn_v4, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 3);

TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rna, asc_bfloat162e1m2x2_rna, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 4);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rna_v1, asc_bfloat162e1m2x2_rna_v2, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 5);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rna_v2, asc_bfloat162e1m2x2_rna_v3, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 6);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rna_v3, asc_bfloat162e1m2x2_rna_v4, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 7);

TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rd, asc_bfloat162e1m2x2_rd, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 8);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rd_v1, asc_bfloat162e1m2x2_rd_v2, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 9);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rd_v2, asc_bfloat162e1m2x2_rd_v3, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 10);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rd_v3, asc_bfloat162e1m2x2_rd_v4, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 11);

TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_ru, asc_bfloat162e1m2x2_ru, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 12);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_ru_v1, asc_bfloat162e1m2x2_ru_v2, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 13);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_ru_v2, asc_bfloat162e1m2x2_ru_v3, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 14);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_ru_v3, asc_bfloat162e1m2x2_ru_v4, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 15);

TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rz, asc_bfloat162e1m2x2_rz, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 16);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rz_v1, asc_bfloat162e1m2x2_rz_v2, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 17);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rz_v2, asc_bfloat162e1m2x2_rz_v3, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 18);
TEST_VECTOR_COMPUTE_VCVT_INSTR_4(
    Bfloat162e1m2x2_rz_v3, asc_bfloat162e1m2x2_rz_v4, vcvt, vector_f4e1m2x2, vector_bfloat16_t, 19);
