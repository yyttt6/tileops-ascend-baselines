/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_storealign_repeat_instr_utils.h"

TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, int8_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, uint8_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, int16_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, uint16_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, int32_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, uint32_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, bfloat16_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, fp8_e4m3fn_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR_HIF8(Vsstb, asc_storealign, vsstb, hifloat8_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, fp8_e5m2_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, fp8_e8m0_t, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, half, 0);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_INSTR(Vsstb, asc_storealign, vsstb, float, 0);

TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, int8_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, uint8_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, int16_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, uint16_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, int32_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, uint32_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, bfloat16_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, fp8_e4m3fn_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR_HIF8(VsstbPost, asc_storealign_postupdate, vsstb, hifloat8_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, fp8_e5m2_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, fp8_e8m0_t, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, half, 1);
TEST_VECTOR_DATAMOVE_STOREALIGN_REPEAT_POST_INSTR(VsstbPost, asc_storealign_postupdate, vsstb, float, 1);
