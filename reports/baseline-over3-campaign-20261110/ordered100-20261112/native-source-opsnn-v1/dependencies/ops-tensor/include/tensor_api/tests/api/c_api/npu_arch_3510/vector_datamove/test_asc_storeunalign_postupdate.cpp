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

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0_HIF8(Vstus, asc_storeunalign_postupdate, vstus, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0(Vstus, asc_storeunalign_postupdate, vstus, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_0_INT4B(Vstus, asc_storeunalign_postupdate, vstus, int4b_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1_HIF8(Vstu, asc_storeunalign_postupdate, vstu, hifloat8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1(Vstu, asc_storeunalign_postupdate, vstu, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_1_INT4B(Vstu, asc_storeunalign_postupdate, vstu, int4b_t);

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, int8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, uint8_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, int16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, int32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, int64_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, half);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, float);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, bfloat16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6(Vstur, asc_storeunalign_postupdate, vstur, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_6_INT4B(Vstur, asc_storeunalign_postupdate, vstur, int4b_t);

TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_PSTU(Pstu0, asc_storeunalign_postupdate, pstu, uint8_t, uint32_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_PSTU(Pstu0, asc_storeunalign_postupdate, pstu, uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_STOREUNALIGN_INSTR_PSTU(Pstu0, asc_storeunalign_postupdate, pstu, uint32_t, uint32_t);
