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

//================asc_int162half_rd================
TEST_VECTOR_COMPUTE_VCVT_INSTR_2(Vcvt_rd, asc_int162half_rd, vcvt, vector_half, vector_int16_t, 0);

//================asc_int162half_rn================
TEST_VECTOR_COMPUTE_VCVT_INSTR_2(Vcvt_rn, asc_int162half_rn, vcvt, vector_half, vector_int16_t, 1);

//================asc_int162half_rna================
TEST_VECTOR_COMPUTE_VCVT_INSTR_2(Vcvt_rna, asc_int162half_rna, vcvt, vector_half, vector_int16_t, 2);

//================asc_int162half_ru================
TEST_VECTOR_COMPUTE_VCVT_INSTR_2(Vcvt_ru, asc_int162half_ru, vcvt, vector_half, vector_int16_t, 3);

//================asc_int162half_rz================
TEST_VECTOR_COMPUTE_VCVT_INSTR_2(Vcvt_rz, asc_int162half_rz, vcvt, vector_half, vector_int16_t, 4);
