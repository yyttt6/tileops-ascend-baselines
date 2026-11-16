/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_storeunalign_instr_utils.h"

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2_HIF8(Vstus0, asc_storeunalign, vstus, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2(Vstus0, asc_storeunalign, vstus, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_2_INT4B(Vstus0, asc_storeunalign, vstus, int4b_t);
