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

TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rd, asc_half2int4x2_rd, vcvt_f162s4, vector_int4x2_t, vector_half, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rd_sat, asc_half2int4x2_rd_sat, vcvt_f162s4, vector_int4x2_t, vector_half, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rd_v2, asc_half2int4x2_rd_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rd_sat_v2, asc_half2int4x2_rd_sat_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 3);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rd_v3, asc_half2int4x2_rd_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 4);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rd_sat_v3, asc_half2int4x2_rd_sat_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 5);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rd_v4, asc_half2int4x2_rd_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 6);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rd_sat_v4, asc_half2int4x2_rd_sat_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 7);

TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rz, asc_half2int4x2_rz, vcvt_f162s4, vector_int4x2_t, vector_half, 10);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rz_sat, asc_half2int4x2_rz_sat, vcvt_f162s4, vector_int4x2_t, vector_half, 11);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rz_v2, asc_half2int4x2_rz_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 12);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rz_sat_v2, asc_half2int4x2_rz_sat_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 13);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rz_v3, asc_half2int4x2_rz_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 14);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rz_sat_v3, asc_half2int4x2_rz_sat_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 15);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rz_v4, asc_half2int4x2_rz_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 16);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rz_sat_v4, asc_half2int4x2_rz_sat_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 17);

TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rn, asc_half2int4x2_rn, vcvt_f162s4, vector_int4x2_t, vector_half, 20);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rn_sat, asc_half2int4x2_rn_sat, vcvt_f162s4, vector_int4x2_t, vector_half, 21);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rn_v2, asc_half2int4x2_rn_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 22);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rn_sat_v2, asc_half2int4x2_rn_sat_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 23);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rn_v3, asc_half2int4x2_rn_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 24);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rn_sat_v3, asc_half2int4x2_rn_sat_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 25);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rn_v4, asc_half2int4x2_rn_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 26);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rn_sat_v4, asc_half2int4x2_rn_sat_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 27);

TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rna, asc_half2int4x2_rna, vcvt_f162s4, vector_int4x2_t, vector_half, 30);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rna_sat, asc_half2int4x2_rna_sat, vcvt_f162s4, vector_int4x2_t, vector_half, 31);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rna_v2, asc_half2int4x2_rna_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 32);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rna_sat_v2, asc_half2int4x2_rna_sat_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 33);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rna_v3, asc_half2int4x2_rna_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 34);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rna_sat_v3, asc_half2int4x2_rna_sat_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 35);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_rna_v4, asc_half2int4x2_rna_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 36);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_rna_sat_v4, asc_half2int4x2_rna_sat_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 37);

TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_ru, asc_half2int4x2_ru, vcvt_f162s4, vector_int4x2_t, vector_half, 40);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_ru_sat, asc_half2int4x2_ru_sat, vcvt_f162s4, vector_int4x2_t, vector_half, 41);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_ru_v2, asc_half2int4x2_ru_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 42);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_ru_sat_v2, asc_half2int4x2_ru_sat_v2, vcvt_f162s4, vector_int4x2_t, vector_half, 43);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_ru_v3, asc_half2int4x2_ru_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 44);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_ru_sat_v3, asc_half2int4x2_ru_sat_v3, vcvt_f162s4, vector_int4x2_t, vector_half, 45);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(vcvt_ru_v4, asc_half2int4x2_ru_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 46);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(
    vcvt_ru_sat_v4, asc_half2int4x2_ru_sat_v4, vcvt_f162s4, vector_int4x2_t, vector_half, 47);
