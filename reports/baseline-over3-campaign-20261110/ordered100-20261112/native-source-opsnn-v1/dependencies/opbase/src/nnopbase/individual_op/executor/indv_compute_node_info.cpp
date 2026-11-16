/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_compute_node_info.h"
#include "securec.h"
#include "exe_graph/runtime/continuous_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus NnopbaseComputeNodeAttrsUpdt(NnopbaseComputeNodeInfoExt* nodeExt, NnopbaseAttrs* attrs)
{
    NnopbaseRuntimeAttrsDef* attrDef = nodeExt->attrStart;
    NnopbaseUChar* ptr = op::internal::PtrCastTo<NnopbaseUChar>(attrDef);
    size_t current_offset = sizeof(NnopbaseRuntimeAttrsDef) + sizeof(size_t) * attrs->num;
    attrDef->attr_num = attrs->num;
    for (size_t i = 0U; i < attrDef->attr_num; i++) {
        attrDef->offset[i] = current_offset;
        if (attrs->attrs[i].addr.isVector) {
            const size_t totalSize = sizeof(gert::ContinuousVector) + attrs->attrs[i].addr.size;
            gert::ContinuousVector* const vector = op::internal::PtrCastTo<gert::ContinuousVector>(ptr +
                                                                                                   current_offset);
            const size_t capacity = attrs->attrs[i].addr.size / attrs->attrs[i].addr.elementSize;
            const size_t size = attrs->attrs[i].addr.size / attrs->attrs[i].addr.elementSize;
            vector->Init(capacity);
            const auto ret = vector->SetSize(size);
            NNOPBASE_ASSERT_TRUE_RETVAL(ret == ge::GRAPH_SUCCESS);
            const auto memRet = memcpy_s(vector->MutableData(), attrs->attrs[i].addr.size, attrs->attrs[i].addr.addr,
                                         attrs->attrs[i].addr.size);
            CHECK_COND(memRet == EOK, ACLNN_ERR_PARAM_INVALID,
                       "Failed to execute memcpy_s! ret = %d, src = %p, dst = %p, len = %zu.", memRet,
                       attrs->attrs[i].addr.addr, vector->MutableData(), attrs->attrs[i].addr.size);
            current_offset += totalSize;
        } else {
            const auto ret = memcpy_s(ptr + current_offset, attrs->attrs[i].addr.size, attrs->attrs[i].addr.addr,
                                      attrs->attrs[i].addr.size);
            CHECK_COND(ret == EOK, ACLNN_ERR_PARAM_INVALID,
                       "Failed to execute memcpy_s! ret = %d, src = %p, dst = %p, len = %zu.", ret,
                       attrs->attrs[i].addr.addr, ptr + current_offset, attrs->attrs[i].addr.size);
            current_offset += attrs->attrs[i].addr.size;
        }
    }
    return OK;
}

aclnnStatus NnopbaseComputeNodeInfoUpdt(NnopbaseExecutor* executor)
{
    NnopbaseComputeNodeInfoExt* nodeExt = &executor->tiling.contextExt.nodeExt;
    NnopbaseComputeNodeInfo* node = nodeExt->node;
    const size_t count = static_cast<size_t>(executor->args->inputs.paramDescs.count);
    const size_t outputIrNum = static_cast<size_t>(executor->args->outputs.paramDescs.count);

    if (!executor->tiling.contextExt.hasPrepared) {
        node->nodeType = executor->opType;
        node->nodeName = executor->opType;
        node->irInputsNum = count;
        node->irOutputsNum = outputIrNum;
    }
    node->inputsNum = executor->args->inputs.num;
    node->outputsNum = executor->args->outputs.num;
    node->attrSize = sizeof(NnopbaseRuntimeAttrsDef) + executor->attrs.totalSize + sizeof(size_t) * executor->attrs.num;
    if ((executor->attrs.num != 0U) || executor->args->inputs.hasDynamic || executor->args->outputs.hasDynamic) {
        const size_t size = NnopbaseComputeNodeCalcLen(executor);
        if (size + sizeof(NnopbaseComputeNodeInfo) > nodeExt->bufLen) {
            node = (NnopbaseComputeNodeInfo*)malloc(sizeof(NnopbaseComputeNodeInfo) + size);
            NNOPBASE_ASSERT_NOTNULL_RETVAL(node);
            nodeExt->node = node;
            auto ret = memcpy_s(node, sizeof(NnopbaseComputeNodeInfo), nodeExt->buf, sizeof(NnopbaseComputeNodeInfo));
            // 异常分支node随nodeExt，跟随executor销毁时释放
            CHECK_COND(ret == EOK, ACLNN_ERR_PARAM_INVALID,
                       "Failed to execute memcpy_s! ret = %d, src = %p, dst = %p, len = %zu.", ret, nodeExt->buf, node,
                       sizeof(NnopbaseComputeNodeInfo));

            if (nodeExt->buf != nullptr) {
                free(nodeExt->buf);
            }
            nodeExt->buf = op::internal::PtrCastTo<NnopbaseUChar>(node);
            nodeExt->bufLen = sizeof(NnopbaseComputeNodeInfo) + size;
            nodeExt->instStart = op::internal::PtrCastTo<NnopbaseAnchorInstanceInfo>(&(node->place_holder));
        }
    }
    nodeExt->inputTdStart = op::internal::PtrCastTo<NnopbaseCompileTimeTensorDesc>(nodeExt->instStart + count);
    nodeExt->outputTdStart = nodeExt->inputTdStart + node->inputsNum;
    nodeExt->attrStart = op::internal::PtrCastTo<NnopbaseRuntimeAttrsDef>(nodeExt->outputTdStart + node->outputsNum);
    nodeExt->outputInstStart = op::internal::PtrCastTo<NnopbaseAnchorInstanceInfo>(
        op::internal::PtrCastTo<NnopbaseUChar>(nodeExt->attrStart) + nodeExt->node->attrSize);

    const NnopbaseParamDesc* const desc = &executor->args->inputs.paramDescs;
    for (size_t i = 0U; i < count; i++) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseComputeNodeSetInstInfo(
            nodeExt->node->irInputsNum, i, nodeExt->instStart, desc->instances[i].startIndex, desc->instances[i].num));
    }
    const NnopbaseParamDesc* const outputDesc = &executor->args->outputs.paramDescs;
    for (size_t i = 0U; i < outputIrNum; i++) {
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseComputeNodeSetInstInfo(nodeExt->node->irOutputsNum, i, nodeExt->outputInstStart,
                                           outputDesc->instances[i].startIndex, outputDesc->instances[i].num));
    }
    return OK;
}

aclnnStatus NnopbaseComputeNodeInfoInit(NnopbaseComputeNodeInfoExt* nodeExt)
{
    nodeExt->node = (NnopbaseComputeNodeInfo*)malloc(sizeof(NnopbaseComputeNodeInfo) + NNOPBASE_COMPUTE_NODE_BUF_LEN);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nodeExt->node);
    nodeExt->node->nodeType = nullptr;
    nodeExt->node->nodeName = nullptr;
    nodeExt->node->irInputsNum = 0U;
    nodeExt->node->inputsNum = 0U;
    nodeExt->node->outputsNum = 0U;
    nodeExt->node->irOutputsNum = 0U;
    nodeExt->node->attrSize = 0U;
    nodeExt->inputTdStart = nullptr;
    nodeExt->outputTdStart = nullptr;
    nodeExt->attrStart = nullptr;
    nodeExt->buf = op::internal::PtrCastTo<NnopbaseUChar>(nodeExt->node);
    nodeExt->bufLen = sizeof(NnopbaseComputeNodeInfo) + NNOPBASE_COMPUTE_NODE_BUF_LEN;
    nodeExt->instStart = op::internal::PtrCastTo<NnopbaseAnchorInstanceInfo>(&(nodeExt->node->place_holder));
    nodeExt->outputInstStart = nullptr;
    return OK;
}

aclnnStatus NnopbaseTilingContextUpdtPrepare(NnopbaseExecutor* executor)
{
    NnopbaseKernelRunContext* const context = executor->tiling.contextExt.context;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseComputeNodeInfoUpdt(executor));
    context->input_size = static_cast<size_t>(executor->args->inputs.num + executor->args->outputs.num +
                                              static_cast<uint32_t>(kInputsAppendEnd));
    context->output_size = gert::TilingContext::kOutputNum;
    context->output_start = &context->values[context->input_size];
    context->compute_node_info = executor->tiling.contextExt.nodeExt.node;
    return OK;
}

#ifdef __cplusplus
}
#endif
