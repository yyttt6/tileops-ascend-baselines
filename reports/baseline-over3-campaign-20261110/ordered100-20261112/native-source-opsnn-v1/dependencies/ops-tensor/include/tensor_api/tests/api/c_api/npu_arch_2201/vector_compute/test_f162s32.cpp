/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/test_conv_instr_utils.h"

// ==========asc_half2int32(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32a, asc_half2int32_rna, vconv_f162s32a, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32c, asc_half2int32_ru, vconv_f162s32c, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32f, asc_half2int32_rd, vconv_f162s32f, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32r, asc_half2int32_rn, vconv_f162s32r, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32z, asc_half2int32_rz, vconv_f162s32z, int32_t, half);
