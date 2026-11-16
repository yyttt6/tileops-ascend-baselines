/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_node.h"

namespace op::mem {
KernelNode::KernelNode(uint32_t opType) : opType_(opType), topoId_(0), originalId_(0) {}

KernelNode::KernelNode(uint32_t opType, uint32_t originalId)
    : opType_(opType), topoId_(originalId), originalId_(originalId)
{}

KernelNode::~KernelNode()
{
    for (auto& input : inputs_) {
        if (input != nullptr) {
            delete input;
            input = nullptr;
        }
    }
    for (auto& output : outputs_) {
        if (output != nullptr) {
            delete output;
            output = nullptr;
        }
    }
    for (auto& wsp : workspace_) {
        if (wsp != nullptr) {
            delete wsp;
            wsp = nullptr;
        }
    }
}

void KernelNode::SetOriginalId(int64_t originalId) { originalId_ = originalId; }

void KernelNode::AddInput(KernelTensor* input)
{
    input->SetOwnerNode(this);
    inputs_.emplace_back(input);
}

void KernelNode::AddOutput(KernelTensor* output)
{
    output->SetOwnerNode(this);
    outputs_.emplace_back(output);
}

void KernelNode::AddWorkspace(KernelTensor* workspace)
{
    workspace->SetOwnerNode(this);
    workspace_.emplace_back(workspace);
}

void KernelNode::SetInputs(op::FVector<KernelTensor*, BASIC_NUM>& inputs)
{
    inputs_ = std::move(inputs);
    for (const auto& input : inputs_) {
        input->SetOwnerNode(this);
    }
}

void KernelNode::SetOutputs(op::FVector<KernelTensor*, BASIC_NUM>& outputs)
{
    outputs_ = std::move(outputs);
    for (const auto& output : outputs_) {
        output->SetOwnerNode(this);
    }
}

void KernelNode::SetWorkspace(op::FVector<KernelTensor*, BASIC_NUM>& workspace)
{
    workspace_ = std::move(workspace);
    for (const auto& wsp : workspace_) {
        wsp->SetOwnerNode(this);
    }
}

const op::FVector<KernelTensor*, BASIC_NUM>& KernelNode::GetInputs() const { return inputs_; }

KernelTensor* KernelNode::GetInput(uint32_t index) const
{
    if (index >= inputs_.size()) {
        return nullptr;
    }
    return inputs_[index];
}

KernelTensor* KernelNode::GetOutput(uint32_t index) const
{
    if (index >= outputs_.size()) {
        return nullptr;
    }
    return outputs_[index];
}

KernelTensor* KernelNode::GetWorkspace(uint32_t index) const
{
    if (index >= workspace_.size()) {
        return nullptr;
    }
    return workspace_[index];
}

KernelGraph* KernelNode::GetOwnerGraph() const { return ownerGraph_; }
} // namespace op::mem
