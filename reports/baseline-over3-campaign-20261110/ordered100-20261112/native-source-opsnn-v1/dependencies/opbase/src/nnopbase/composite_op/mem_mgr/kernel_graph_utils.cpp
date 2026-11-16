/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../common/inc/kernel_graph_utils.h"
#include "opdev/op_log.h"

namespace op::mem {
aclnnStatus KernelGraphUtils::Link(const KernelNode* fromNode, const KernelNode* toNode, size_t from, size_t to)
{
    if (fromNode == nullptr || toNode == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "fromNode or toNode is nullptr");
        return ACLNN_ERR_INNER;
    }

    auto& fromNodeOutputs = fromNode->GetOutputs();
    if (from > fromNodeOutputs.size()) {
        OP_LOGE(ACLNN_ERR_INNER, "from %zu is larger than fromNode output size %zu.", from, fromNodeOutputs.size());
        return ACLNN_ERR_INNER;
    }

    auto& toNodeInputs = toNode->GetInputs();
    if (to > toNodeInputs.size()) {
        OP_LOGE(ACLNN_ERR_INNER, "to %zu is larger than toNode input size %zu.", to, toNodeInputs.size());
        return ACLNN_ERR_INNER;
    }
    auto* fromTensor = fromNodeOutputs.at(from);
    auto* toTensor = toNode->GetInputs().at(to);
    fromTensor->AddPeerTensor(toTensor);
    toTensor->AddPeerTensor(fromTensor, true);
    return ACLNN_SUCCESS;
}
} // namespace op::mem
