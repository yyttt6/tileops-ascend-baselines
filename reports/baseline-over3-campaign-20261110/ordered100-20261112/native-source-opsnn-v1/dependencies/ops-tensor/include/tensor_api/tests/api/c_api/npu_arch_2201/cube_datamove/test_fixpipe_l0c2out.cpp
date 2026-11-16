/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/test_copy_instr_utils.h"

TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, half, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, bfloat16_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, int8_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, uint8_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, float, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, half, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, int16_t, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, int8_t, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2Out, asc_copy_l0c2gm, copy_matrix_cc_to_gm, int32_t, int32_t);

TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, half, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, bfloat16_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, int8_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, uint8_t, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, float, float);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, half, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, int16_t, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, int8_t, int32_t);
TEST_CUBE_COMPUTE_FIXPIPE_INSTR(L0c2OutSync, asc_copy_l0c2gm_sync, copy_matrix_cc_to_gm, int32_t, int32_t);
