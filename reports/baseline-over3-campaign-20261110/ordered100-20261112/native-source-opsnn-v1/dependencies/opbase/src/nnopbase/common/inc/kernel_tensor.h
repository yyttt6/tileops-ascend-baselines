/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_TENSOR_H_
#define OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_TENSOR_H_

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/fast_vector.h"
#include "opdev/object.h"
#include "opdev/op_log.h"
#include <cstdint>

namespace op::mem {
enum class TensorType { INPUT = 0, OUTPUT, WORKSPACE, OUTPUT_REF, USER_INPUT, USER_OUTPUT, DEFAULT };

const size_t BASIC_NUM = 32; // this is the basic number of inputs, outputs, peer inputs and peer outputs.
const size_t DEFAULT_STACK_SIZE = 8;
const size_t DEFAULT_NODE_NUM = 16;
const size_t DEFAULT_TENSOR_NUM = 32;
const size_t BITS_PER_BYTE = 8;
const size_t BITS_PER_BYTE_SHIFT = 3;

class KernelNode;

class KernelTensor;
struct aclTensorExtend : public Object {
public:
    KernelTensor* GetKernelTensor() { return kernelTensor_; }
    void SetKernelTensor(KernelTensor* kernelTensor) { kernelTensor_ = kernelTensor; }
    ~aclTensorExtend() { kernelTensor_ = nullptr; }

private:
    KernelTensor* kernelTensor_{nullptr};
};

class KernelTensor : public op::Object {
public:
    KernelTensor(aclTensor* tensor, uint64_t index);

    ~KernelTensor();

    int64_t GetSize() const { return tensorSize_; }

    aclnnStatus CalcSize()
    {
        CHECK_RET(aclTensor_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

        size_t typeSize = op::TypeSize(aclTensor_->GetDataType());
        if (typeSize > ge::kDataTypeSizeBitOffset) {
            if (ge::MulOverflow((typeSize - ge::kDataTypeSizeBitOffset), aclTensor_->GetStorageShape().GetShapeSize(),
                                tensorSize_)) {
                return ACLNN_ERR_INNER;
            }
            tensorSize_ = (tensorSize_ + (BITS_PER_BYTE - 1)) >> BITS_PER_BYTE_SHIFT;
            return ACLNN_SUCCESS;
        } else {
            if (ge::MulOverflow(typeSize, aclTensor_->GetStorageShape().GetShapeSize(), tensorSize_)) {
                return ACLNN_ERR_INNER;
            } else {
                return ACLNN_SUCCESS;
            }
        }
    }

    void SetSize(int64_t tensorSize) { tensorSize_ = tensorSize; }

    int64_t GetLifeTimeStart() const;

    int64_t GetLifeTimeEnd() const;

    void SetLifeTimeStart(int64_t lifeTimeStart) { lifeTimeStart_ = lifeTimeStart; }

    void SetLifeTimeEnd(int64_t lifeTimeEnd) { lifeTimeEnd_ = lifeTimeEnd; }

    bool IsLifeTimeNotOverLap(const KernelTensor* anotherTensor) const;

    size_t GetIndex() const;

    aclTensor* GetAclTensor() const { return aclTensor_; }

    KernelNode* GetOwnerNode() const { return ownerNode_; }

    void SetOwnerNode(KernelNode* ownerNode);

    void AddPeerTensor(KernelTensor* peerTensor, bool updateAclTensor = false);

    const op::FVector<KernelTensor*, BASIC_NUM>& GetPeerTensors() const { return peerKernelTensors_; }
    void SetOffset(uint64_t offset)
    {
        if (aclTensor_ == nullptr) {
            OP_LOGW("aclTensor is null, cant no set workspace offset.");
            return;
        }
        aclTensor_->SetWorkspaceOffset(offset);
    }

    TensorType GetType() const;

    void SetType(TensorType type);

    void SetRefTensor(KernelTensor* refTensor) { refTensor_ = refTensor; }

    KernelTensor* GetRefTensor() { return refTensor_; }

    bool IsInputOf(const KernelTensor* output);

private:
    aclTensor* aclTensor_;
    op::FVector<KernelTensor*, BASIC_NUM> peerKernelTensors_;
    KernelNode* ownerNode_ = nullptr;
    int64_t tensorSize_{0};
    int64_t index_; // 表示第几个输出、输入
    KernelTensor* refTensor_ = nullptr;
    TensorType type_ = TensorType::DEFAULT;
    int64_t lifeTimeStart_ = 0L;
    int64_t lifeTimeEnd_ = INT64_MAX;
};
} // namespace op::mem

#endif // OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_TENSOR_H_
