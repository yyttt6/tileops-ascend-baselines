/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_COMPUTE_NODE_INFO_H_
#define INDV_COMPUTE_NODE_INFO_H_

#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "indv_executor.h"
#include "exe_graph/runtime/tiling_context.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NNOPBASE_COMPUTE_NODE_BUF_LEN = 4096U;
static constexpr size_t NNOPBASE_COMPUTE_NODE_BUF_LEN_EXT = 128U;

aclnnStatus NnopbaseComputeNodeAttrsUpdt(NnopbaseComputeNodeInfoExt* nodeExt, NnopbaseAttrs* attrs);
aclnnStatus NnopbaseComputeNodeInfoUpdt(NnopbaseExecutor* executor);
aclnnStatus NnopbaseComputeNodeInfoInit(NnopbaseComputeNodeInfoExt* nodeExt);

static inline void NnopbaseComputeNodeInfoDeInit(NnopbaseComputeNodeInfoExt* nodeExt)
{
    if (nodeExt->buf) {
        free(nodeExt->buf);
    }
    nodeExt->buf = nullptr;
    nodeExt->bufLen = 0;
}

static inline size_t NnopbaseComputeNodeCalcLen(NnopbaseExecutor* executor)
{
    NnopbaseComputeNodeInfo* node = executor->tiling.contextExt.nodeExt.node;
    size_t size = node->irInputsNum * sizeof(NnopbaseAnchorInstanceInfo) +
                  node->irOutputsNum * sizeof(NnopbaseAnchorInstanceInfo) +
                  (node->outputsNum + node->inputsNum) * sizeof(NnopbaseCompileTimeTensorDesc) +
                  sizeof(NnopbaseRuntimeAttrsDef) + executor->attrs.totalSize + sizeof(size_t) * executor->attrs.num +
                  NNOPBASE_COMPUTE_NODE_BUF_LEN_EXT;
    return size;
}

static inline aclnnStatus NnopbaseComputeNodeSetInstInfo(const size_t irNum, const size_t irIndex,
                                                         NnopbaseAnchorInstanceInfo* nodeExtInstStart,
                                                         const uint32_t instanceStart, const uint32_t instantiationNum)
{
    NNOPBASE_ASSERT_TRUE_RETVAL(irIndex < irNum);
    NnopbaseAnchorInstanceInfo* inst = nodeExtInstStart + irIndex;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(inst);
    inst->SetInstanceStart(instanceStart);
    inst->SetInstantiationNum(instantiationNum);
    return OK;
}

aclnnStatus NnopbaseTilingContextUpdtPrepare(NnopbaseExecutor* executor);

static inline aclnnStatus NnopbaseTilingBuildOpAttrs(NnopbaseExecutor* executor)
{
    return NnopbaseComputeNodeAttrsUpdt(&executor->tiling.contextExt.nodeExt, &executor->attrs);
}
#ifdef __cplusplus
}
#endif
#endif
