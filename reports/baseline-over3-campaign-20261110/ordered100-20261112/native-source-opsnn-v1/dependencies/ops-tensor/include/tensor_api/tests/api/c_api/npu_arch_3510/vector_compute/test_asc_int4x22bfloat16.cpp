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

TEST_VECTOR_COMPUTE_VCVT_INSTR_3(Vcvt, asc_int4x22bfloat16, vcvt_s42bf16, vector_bfloat16_t, vector_int4x2_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_3(Vcvt_v2, asc_int4x22bfloat16_v2, vcvt_s42bf16, vector_bfloat16_t, vector_int4x2_t, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_3(Vcvt_v3, asc_int4x22bfloat16_v3, vcvt_s42bf16, vector_bfloat16_t, vector_int4x2_t, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_3(Vcvt_v4, asc_int4x22bfloat16_v4, vcvt_s42bf16, vector_bfloat16_t, vector_int4x2_t, 3);
