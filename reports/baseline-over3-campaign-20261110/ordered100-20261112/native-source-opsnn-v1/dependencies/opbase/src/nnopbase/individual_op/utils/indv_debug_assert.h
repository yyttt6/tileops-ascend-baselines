/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_COMMON_ASSERT_H_
#define INDV_COMMON_ASSERT_H_

#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "profiling/prof_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NNOPBASE_ASSERT(exp)                                    \
    do {                                                        \
        if (!(exp)) {                                           \
            OP_LOGE(ACLNN_ERR_INNER, "Assert %s failed", #exp); \
            return;                                             \
        }                                                       \
    } while (false)

#define NNOPBASE_ASSERT_RETVAL(exp, ret)            \
    do {                                            \
        if (!(exp)) {                               \
            OP_LOGE(ret, "Assert %s failed", #exp); \
            return (ret);                           \
        }                                           \
    } while (false)

#define NNOPBASE_ASSERT_OK_RETVAL(v)                     \
    do {                                                 \
        const aclnnStatus _chk_status = (v);             \
        if (_chk_status != OK) {                         \
            OP_LOGE(_chk_status, "Check %s failed", #v); \
            return (_chk_status);                        \
        }                                                \
    } while (false)

#define NNOPBASE_ASSERT_RTOK_RETVAL(v) NNOPBASE_ASSERT_RETVAL(((v) == 0), (ACLNN_ERR_RUNTIME_ERROR))
#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v) NNOPBASE_ASSERT_RETVAL(((v) != nullptr), (ACLNN_ERR_PARAM_NULLPTR))
#define NNOPBASE_ASSERT_TRUE_RETVAL(v) NNOPBASE_ASSERT_RETVAL((v), (ACLNN_ERR_PARAM_INVALID))
#define NNOPBASE_MSPROF_OK_RETVAL(v) NNOPBASE_ASSERT_RETVAL(((v) == MSPROF_ERROR_NONE), (ACLNN_ERR_PARAM_INVALID))

#define NNOPBASE_ASSERT_OK(v) NNOPBASE_ASSERT(((v) == OK))
#define NNOPBASE_ASSERT_RTOK(v) NNOPBASE_ASSERT(((v) == 0))
#define NNOPBASE_ASSERT_NOTNULL(v) NNOPBASE_ASSERT(((v) != nullptr))
#define NNOPBASE_ASSERT_TRUE(v) NNOPBASE_ASSERT((v))

#ifdef __cplusplus
}
#endif
#endif
