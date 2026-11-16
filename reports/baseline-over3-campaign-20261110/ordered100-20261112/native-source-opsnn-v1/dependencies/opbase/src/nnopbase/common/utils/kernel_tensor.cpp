/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tensor.h"
#include "kernel_node.h"

namespace op::mem {
class KernelNode;

KernelTensor::KernelTensor(aclTensor* tensor, uint64_t index) : aclTensor_(tensor), index_(index) {}

KernelTensor::~KernelTensor()
{
    if (aclTensor_ != nullptr) {
        aclTensorExtend* extendTensor = (aclTensorExtend*)aclTensor_->GetExtend();
        if (extendTensor != nullptr && extendTensor->GetKernelTensor() == this) {
            OP_LOGD("~KernelTensor set extendTensor %p to null", extendTensor);
            extendTensor->SetKernelTensor(nullptr);
        }
        aclTensor_ = nullptr;
    }
}

int64_t KernelTensor::GetLifeTimeStart() const { return lifeTimeStart_; }

int64_t KernelTensor::GetLifeTimeEnd() const { return lifeTimeEnd_; }

bool KernelTensor::IsLifeTimeNotOverLap(const KernelTensor* anotherTensor) const
{
    if (anotherTensor->lifeTimeStart_ == lifeTimeStart_) {
        /* Twp tensors have same lifetime start means they are
         * from same operator and Their memory must be different.*/
        return false;
    }

    if (anotherTensor->lifeTimeStart_ > lifeTimeStart_) {
        return lifeTimeEnd_ < anotherTensor->lifeTimeStart_;
    } else {
        return anotherTensor->lifeTimeEnd_ < lifeTimeStart_;
    }
}

size_t KernelTensor::GetIndex() const { return index_; }

void KernelTensor::SetOwnerNode(KernelNode* ownerNode) { ownerNode_ = ownerNode; }

void KernelTensor::AddPeerTensor(KernelTensor* peerTensor, bool updateAclTensor)
{
    peerKernelTensors_.emplace_back(peerTensor);
    if (updateAclTensor) {
        aclTensor_ = peerTensor->aclTensor_;
    }
}

TensorType KernelTensor::GetType() const { return type_; }

void KernelTensor::SetType(TensorType type) { type_ = type; }

bool KernelTensor::IsInputOf(const KernelTensor* output)
{
    KernelNode* owner = output->GetOwnerNode();
    if (owner) {
        auto& inputs = owner->GetInputs();
        for (auto input : inputs) {
            if (input->GetAclTensor() == GetAclTensor()) {
                return true;
            }
        }
    }
    return false;
}
} // namespace op::mem