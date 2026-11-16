/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_MC2_AICPU_H_
#define INDV_MC2_AICPU_H_
#include "indv_executor.h"
#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus NnopbaseGetHcomResource(NnopbaseExecutor* executor, aclrtStream const stream);
aclnnStatus NnopbaseAicpuKernelLaunch(NnopbaseExecutor* const executor);
aclnnStatus NnopbasePrepareMC2Params(NnopbaseExecutor* executor, NnopbaseExecutorArgsAddr* argsAddr);
aclnnStatus NnopbaseLaunchKFCTask(NnopbaseExecutor* const executor, aclrtStream stream);
aclnnStatus NnopbaseLaunchKFCTaskA5(NnopbaseExecutor* const executor, aclrtStream stream);
aclnnStatus NnopbaseExecutorGetMc2Num(NnopbaseExecutor* const executor, aclrtStream const stream,
                                      NnopbaseExecutorArgsAddr* argsAddr, uint32_t* mc2Num);
aclnnStatus NnopbaseFusionKernelLaunch(NnopbaseExecutor* const executor, aclrtStream const stream);
aclnnStatus NnopbaseMC2KernelLaunch(NnopbaseExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
