/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_api/npu_arch_2201/utils/test_conv_instr_utils.h"

TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16, asc_int162half, vconv_s162f16, half, int16_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16a, asc_int162half_rna, vconv_s162f16a, half, int16_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16c, asc_int162half_ru, vconv_s162f16c, half, int16_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16f, asc_int162half_rd, vconv_s162f16f, half, int16_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16r, asc_int162half_rn, vconv_s162f16r, half, int16_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvS162fp16z, asc_int162half_rz, vconv_s162f16z, half, int16_t);
