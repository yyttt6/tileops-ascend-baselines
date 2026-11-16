/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_NODE_H
#define OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_NODE_H
#include <vector>
#include "kernel_tensor.h"
#include "opdev/object.h"
#include "opdev/op_def.h"

namespace op::mem {
class KernelGraph;

class KernelNode : public op::Object {
public:
    explicit KernelNode(uint32_t opType);
    KernelNode(uint32_t opType, uint32_t originalId);
    ~KernelNode();

    int64_t GetTopoId() const { return topoId_; }

    void SetTopoId(int64_t topoId) { topoId_ = topoId; }

    int64_t GetOriginalId() const { return originalId_; }

    uint32_t GetOpType() const { return opType_; }

    void SetOriginalId(int64_t originalId);

    void AddInput(KernelTensor* input);

    void AddOutput(KernelTensor* output);

    void AddWorkspace(KernelTensor* workspace);

    void SetInputs(op::FVector<KernelTensor*, BASIC_NUM>& inputs);

    void SetOutputs(op::FVector<KernelTensor*, BASIC_NUM>& outputs);

    void SetWorkspace(op::FVector<KernelTensor*, BASIC_NUM>& workspace);

    const op::FVector<KernelTensor*, BASIC_NUM>& GetInputs() const;

    const op::FVector<KernelTensor*, BASIC_NUM>& GetOutputs() const { return outputs_; }

    const op::FVector<KernelTensor*, BASIC_NUM>& GetWorkspace() const { return workspace_; }

    KernelTensor* GetInput(uint32_t index) const;

    KernelTensor* GetOutput(uint32_t index) const;

    KernelTensor* GetWorkspace(uint32_t index) const;

    KernelGraph* GetOwnerGraph() const;

private:
    uint32_t opType_;
    int64_t topoId_;
    int64_t originalId_;
    KernelGraph* ownerGraph_ = nullptr;
    op::FVector<KernelTensor*, BASIC_NUM> inputs_;
    op::FVector<KernelTensor*, BASIC_NUM> outputs_;
    op::FVector<KernelTensor*, BASIC_NUM> workspace_;
};
} // namespace op::mem
#endif // OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_NODE_H
