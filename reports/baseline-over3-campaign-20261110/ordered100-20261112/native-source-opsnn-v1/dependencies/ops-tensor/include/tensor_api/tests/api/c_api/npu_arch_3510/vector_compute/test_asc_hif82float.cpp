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

TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Hif82float, asc_hif82float, vcvt, vector_float, vector_hifloat8_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Hif82float_v2, asc_hif82float_v2, vcvt, vector_float, vector_hifloat8_t, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Hif82float_v3, asc_hif82float_v3, vcvt, vector_float, vector_hifloat8_t, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Hif82float_v4, asc_hif82float_v4, vcvt, vector_float, vector_hifloat8_t, 3);
