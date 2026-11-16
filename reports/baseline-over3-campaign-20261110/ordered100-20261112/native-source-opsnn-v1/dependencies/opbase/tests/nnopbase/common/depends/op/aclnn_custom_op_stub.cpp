/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_custom_op_stub.h"
#include <string.h>
#include "graph/types.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_soc.h"
#include "executor/indv_collector.h"
#include "executor/indv_executor.h"
#include "op_stub.h"

namespace {
uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
TensorDesc inputDesc1[3] = {
    {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list1 = {inputDesc1, 3, outputDesc1, 1};
SupportInfo supportInfo0[1] = {list1};
SupportInfo supportInfo1[1] = {list1};
OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
OpSupportList supportList = {opSocSupportList, 2};
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCustomOp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    return NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize);
}

aclnnStatus aclnnCustomOpGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                          const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnCustomOpTiling", false};

    static void* executorSpace = nullptr;
    const char* opType = "custom_op";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(x1);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(x2);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(x3);
    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    void* nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char),
                                             outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                                             sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor*>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    NnopbaseAddOpTypeId(*executor, 32);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x2, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x3, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, out, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, 2));
    aclnnStatus ret = NnopbaseRunForWorkspace(*executor, workspaceSize);
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

#ifdef __cplusplus
}
#endif