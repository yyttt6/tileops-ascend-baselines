/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_context_holder.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "kernel_utils.h"
#include "opdev/op_errno.h"

namespace op::internal {

namespace {
// Growth factor for capacity expansion (similar to std::vector)
constexpr size_t GROWTH_FACTOR = 2;
} // anonymous namespace

void KernelContextHolder::BuildComputeNodeInfo()
{
    computeNodeInfoSize_ = sizeof(ComputeNodeInfo) + sizeof(AnchorInstanceInfo) * computeNodeInfoCapacity_ +
                           sizeof(CompileTimeTensorDesc) * computeNodeInfoCapacity_ + sizeof(RuntimeAttrsDef) +
                           attrCapacity_;
    computeNodeInfo_ = static_cast<ComputeNodeInfo*>(malloc(computeNodeInfoSize_));
    if (computeNodeInfo_ == nullptr) {
        return;
    }
    (void)memset_s(computeNodeInfo_, computeNodeInfoSize_, 0, computeNodeInfoSize_);
    anchorInfo_ = PtrCastTo<AnchorInstanceInfo>(&computeNodeInfo_->place_holder);
}

void KernelContextHolder::BuildOpInOutArg()
{
    size_t sz = sizeof(AsyncAnyValue) * opInArgCapacity_;
    opInArg_ = static_cast<AsyncAnyValue*>(malloc(sz));
    if (opInArg_ == nullptr) {
        return;
    }
    (void)memset_s(opInArg_, sz, 0, sz);
}

aclnnStatus KernelContextHolder::EnsureArgCapacity(size_t requiredCapacity)
{
    if (requiredCapacity <= opInArgCapacity_) {
        return ACLNN_SUCCESS;
    }

    // Calculate new capacity using growth factor 2x (similar to std::vector)
    const size_t oldCapacity = opInArgCapacity_;
    size_t newCapacity = oldCapacity;
    while (newCapacity < requiredCapacity) {
        newCapacity *= GROWTH_FACTOR;
    }

    // Allocate new memory (do NOT use realloc)
    size_t newSize = sizeof(AsyncAnyValue) * newCapacity;
    AsyncAnyValue* newArg = static_cast<AsyncAnyValue*>(malloc(newSize));
    OP_CHECK(newArg != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc opInArg, size %zu.", newSize),
             return ACLNN_ERR_INNER);

    // Copy existing data to new location
    size_t oldSize = sizeof(AsyncAnyValue) * opInArgCapacity_;
    OP_CHECK(memcpy_s(newArg, newSize, opInArg_, oldSize) == EOK, OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy opInArg."),
             std::free(newArg);
             return ACLNN_ERR_INNER);

    // Zero out the new portion
    size_t zeroSize = newSize - oldSize;
    (void)memset_s(PtrCastTo<uint8_t>(newArg) + oldSize, zeroSize, 0, zeroSize);

    // Free old memory and update pointer
    std::free(opInArg_);
    opInArg_ = newArg;
    opInArgCapacity_ = newCapacity;

    OP_LOGI("Expanded opInArg capacity from %zu to %zu.", oldCapacity, opInArgCapacity_);
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::EnsureComputeNodeInfoCapacity(size_t requiredCapacity)
{
    if (requiredCapacity <= computeNodeInfoCapacity_) {
        return ACLNN_SUCCESS;
    }

    // Calculate new capacity using growth factor 2x
    const size_t oldCapacity = computeNodeInfoCapacity_;
    size_t newCapacity = oldCapacity;
    while (newCapacity < requiredCapacity) {
        newCapacity *= GROWTH_FACTOR;
    }

    // Allocate new memory
    size_t newSize = sizeof(ComputeNodeInfo) + sizeof(AnchorInstanceInfo) * newCapacity +
                     sizeof(CompileTimeTensorDesc) * newCapacity + sizeof(RuntimeAttrsDef) + attrCapacity_;
    ComputeNodeInfo* newInfo = static_cast<ComputeNodeInfo*>(malloc(newSize));
    OP_CHECK(newInfo != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc computeNodeInfo, size %zu.", newSize),
             return ACLNN_ERR_INNER);

    // Copy existing data
    OP_CHECK(memcpy_s(newInfo, newSize, computeNodeInfo_, computeNodeInfoSize_) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy computeNodeInfo."), std::free(newInfo);
             return ACLNN_ERR_INNER);

    // Zero out the new portion
    size_t zeroOffset = computeNodeInfoSize_;
    size_t zeroSize = newSize - computeNodeInfoSize_;
    (void)memset_s(PtrCastTo<uint8_t>(newInfo) + zeroOffset, zeroSize, 0, zeroSize);

    // Update pointers based on new memory layout
    anchorInfo_ = PtrCastTo<AnchorInstanceInfo>(&newInfo->place_holder);

    // Recalculate compileDesc_ and attrDef_ based on current irInputNum_
    UpdateCompileDescOffset(irInputNum_);
    if (attrNum_ > 0) {
        UpdateAttrDefOffset(inputNum_ + outputNum_, attrNum_);
    }

    // Note: attrDef offset data is already copied as part of the memcpy_s(newInfo, ...) above

    // Free old memory and update
    std::free(computeNodeInfo_);
    computeNodeInfo_ = newInfo;
    computeNodeInfoSize_ = newSize;
    computeNodeInfoCapacity_ = newCapacity;

    // Update kernelCtx reference in inferShapeCtx and tilingCtx if they exist
    // This is handled by the fact that these contexts hold pointers to the KernelContextHolder,
    // not directly to computeNodeInfo_

    OP_LOGI("Expanded computeNodeInfo capacity from %zu to %zu.", oldCapacity, computeNodeInfoCapacity_);
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::EnsureAttrCapacity(size_t requiredSize)
{
    if (requiredSize <= attrCapacity_) {
        return ACLNN_SUCCESS;
    }

    // Calculate new capacity using growth factor 2x
    const size_t oldCapacity = attrCapacity_;
    size_t newCapacity = oldCapacity;
    while (newCapacity < requiredSize) {
        newCapacity *= GROWTH_FACTOR;
    }

    // Need to reallocate the entire computeNodeInfo_ since attr data is embedded in it
    size_t newSize = sizeof(ComputeNodeInfo) + sizeof(AnchorInstanceInfo) * computeNodeInfoCapacity_ +
                     sizeof(CompileTimeTensorDesc) * computeNodeInfoCapacity_ + sizeof(RuntimeAttrsDef) + newCapacity;
    ComputeNodeInfo* newInfo = static_cast<ComputeNodeInfo*>(malloc(newSize));
    OP_CHECK(newInfo != nullptr,
             OP_LOGE(ACLNN_ERR_INNER, "failed to malloc computeNodeInfo for attr expansion, size %zu.", newSize),
             return ACLNN_ERR_INNER);

    // Copy data up to current computeNodeInfoSize_
    OP_CHECK(memcpy_s(newInfo, newSize, computeNodeInfo_, computeNodeInfoSize_) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy computeNodeInfo."), std::free(newInfo);
             return ACLNN_ERR_INNER);

    // Zero out the new portion
    size_t zeroOffset = computeNodeInfoSize_;
    size_t zeroSize = newSize - computeNodeInfoSize_;
    (void)memset_s(PtrCastTo<uint8_t>(newInfo) + zeroOffset, zeroSize, 0, zeroSize);

    // Update pointers based on new memory layout
    ComputeNodeInfo* oldInfo = computeNodeInfo_;
    computeNodeInfo_ = newInfo;
    anchorInfo_ = PtrCastTo<AnchorInstanceInfo>(&newInfo->place_holder);
    compileDesc_ = PtrCastTo<CompileTimeTensorDesc>(anchorInfo_ + irInputNum_);
    UpdateAttrDefOffset(inputNum_ + outputNum_, attrNum_);

    computeNodeInfoSize_ = newSize;
    attrCapacity_ = newCapacity;

    // Free old memory
    std::free(oldInfo);

    OP_LOGI("Expanded attr capacity from %zu to %zu.", oldCapacity, attrCapacity_);
    return ACLNN_SUCCESS;
}

void KernelContextHolder::UpdateKernelExtendInfo(const char* kernelType, const char* kernelName)
{
    kernelExtendInfo_.kernel_type_ = kernelType;
    kernelExtendInfo_.kernel_name_ = kernelName;
}

void KernelContextHolder::UpdateCompileDescOffset(size_t irInputNum)
{
    compileDesc_ = PtrCastTo<CompileTimeTensorDesc>(anchorInfo_ + irInputNum);
}

void KernelContextHolder::UpdateAttrDefOffset(size_t inoutNum, size_t attrNum)
{
    size_t attrOffset = sizeof(CompileTimeTensorDesc) * inoutNum;
    attrDef_ = PtrCastTo<RuntimeAttrsDef>(PtrShift(compileDesc_, attrOffset));
    attrDef_->attr_num = attrNum;
    attrDataStart_ = PtrCastTo<uint8_t>(attrDef_ + 1) + sizeof(size_t) * attrNum;
#ifdef DEBUG
    OP_LOGD("Adjust attr offset. [%zu]. inout num %zu. attr num: %zu,  %zu", attrOffset, inoutNum, attrNum,
            PtrOffset(computeNodeInfo_, attrDef_));
#endif
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, const aclTensor* tensor)
{
    if (tensor == nullptr) {
        // Caution: RT2.0 seems not support null op input in the middle or op args
        OP_LOGI("Update Input Arg Tensor is NULL: [%zu]", idx);
        anchorInfo_[idx].instance_start_ = inputNum_;
        anchorInfo_[idx].instantiation_num_ = 0;
        return ACLNN_SUCCESS;
    }

#ifdef DEBUG
    OP_LOGD("Update Input Arg Tensor: [%zu]. %s", idx, tensor->ToString().GetString());
    if (tensor->GetPlacement() == gert::kOnHost) {
        OP_LOGD("Update Input arg HOST aclTensor. Size: %zu", tensor->Size());
        for (auto i = 0; i < tensor->Size(); i++) {
            uint64_t* p = (static_cast<uint64_t*>(tensor->GetData()) + i);
            OP_LOGD("Update Input arg HOST aclTensor Data: %lu", *p);
        }
    }
#endif

    // Ensure capacity for input args (need room for one more input)
    CHECK_RET_CODE(EnsureArgCapacity(inputNum_ + 1), "EnsureArgCapacity failed.");
    CHECK_RET_CODE(EnsureComputeNodeInfoCapacity(inputNum_ + 1), "EnsureComputeNodeInfoCapacity failed.");

    anchorInfo_[idx].instance_start_ = inputNum_;
    anchorInfo_[idx].instantiation_num_ = 1;
    compileDesc_[inputNum_].data_type_ = tensor->GetDataType();
    compileDesc_[inputNum_].storage_format_.SetOriginFormat(tensor->GetOriginalFormat());
    compileDesc_[inputNum_].storage_format_.SetStorageFormat(tensor->GetStorageFormat());
    opInArg_[inputNum_].data.pointer = tensor->GetTensor();
    inputNum_++;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, const aclTensorList* tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update Input Arg Tensor List is Null Or Empty");
        anchorInfo_[idx].instance_start_ = inputNum_;
        anchorInfo_[idx].instantiation_num_ = 0;
        return ACLNN_SUCCESS;
    }
    size_t start = inputNum_;
    for (size_t i = 0; i < tensorList->Size(); i++) {
        if (UpdateInputArg(idx, (*tensorList)[i]) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER;
        };
    }
    if (inputNum_ == start) {
        OP_LOGD("Update Input Arg Tensor List contains only NULL tensor");
    }
    anchorInfo_[idx].instance_start_ = start;
    anchorInfo_[idx].instantiation_num_ = inputNum_ - start;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, OpArg& arg)
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            return UpdateInputArg(idx, reinterpret_cast<aclTensor*>(arg->pointer));
        case OpArgType::OPARG_ACLTENSOR_LIST:
            return UpdateInputArg(idx, reinterpret_cast<aclTensorList*>(arg->pointer));
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid input arg type %d.", static_cast<int>(arg.type));
            return ACLNN_ERR_INNER;
    }
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, const aclTensor* tensor)
{
    if (tensor == nullptr) {
        OP_LOGI("Update Output Arg Tensor List is Null [%zu]", idx);
        return ACLNN_SUCCESS;
    }

    // Ensure capacity for input + output args
    size_t totalArgs = inputNum_ + outputNum_ + 1;
    CHECK_RET_CODE(EnsureArgCapacity(totalArgs), "EnsureArgCapacity failed.");
    CHECK_RET_CODE(EnsureComputeNodeInfoCapacity(totalArgs), "EnsureComputeNodeInfoCapacity failed.");

#ifdef DEBUG
    OP_LOGD("Update Output Arg Tensor: [%zu]. %s", idx, tensor->ToString().GetString());
#endif
    compileDesc_[inputNum_ + outputNum_].data_type_ = tensor->GetDataType();
    compileDesc_[inputNum_ + outputNum_].storage_format_.SetOriginFormat(tensor->GetOriginalFormat());
    compileDesc_[inputNum_ + outputNum_].storage_format_.SetStorageFormat(tensor->GetStorageFormat());
    opInArg_[inputNum_ + outputNum_].data.pointer = tensor->GetTensor();
    outputNum_++;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, const aclTensorList* tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update Output Arg Tensor List is Null Or Empty");
        return ACLNN_SUCCESS;
    }
    for (size_t i = 0; i < tensorList->Size(); i++) {
        if (UpdateOutputArg(idx, (*tensorList)[i]) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER;
        };
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, OpArg& arg)
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            return UpdateOutputArg(idx, reinterpret_cast<aclTensor*>(arg->pointer));
        case OpArgType::OPARG_ACLTENSOR_LIST:
            return UpdateOutputArg(idx, reinterpret_cast<aclTensorList*>(arg->pointer));
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid output arg type %d.", static_cast<int>(arg.type));
            return ACLNN_ERR_INNER;
    }
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, const aclTensor* tensor, size_t& seq) const
{
    if (tensor == nullptr) {
        // Caution: RT2.0 seems not support null op input in the middle or op args
        OP_LOGD("Update Input Arg Tensor is NULL: [%zu]", idx);
        outputAnchorInfo_[idx].instance_start_ = seq;
        outputAnchorInfo_[idx].instantiation_num_ = 0;
        return;
    }
    outputAnchorInfo_[idx].instance_start_ = seq++;
    outputAnchorInfo_[idx].instantiation_num_ = 1;
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, const aclTensorList* tensorList, size_t& seq) const
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update IR Output argList is Null Or Empty");
        outputAnchorInfo_[idx].instance_start_ = seq;
        outputAnchorInfo_[idx].instantiation_num_ = 0;
        return;
    }

    size_t start = seq;
    for (size_t i = 0; i < tensorList->Size(); i++) {
        UpdateOutputArgIr(idx, (*tensorList)[i], seq);
    }
    outputAnchorInfo_[idx].instance_start_ = start;
    outputAnchorInfo_[idx].instantiation_num_ = seq - start;
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, OpArg& arg, size_t& seq) const
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            UpdateOutputArgIr(idx, reinterpret_cast<aclTensor*>(arg->pointer), seq);
            break;
        case OpArgType::OPARG_ACLTENSOR_LIST:
            UpdateOutputArgIr(idx, reinterpret_cast<aclTensorList*>(arg->pointer), seq);
            break;
        default:
            OP_LOGW("invalid output arg type %d.", static_cast<int>(arg.type));
            break;
    }
}

void KernelContextHolder::ResetComputeNodeInfo(const char* opType, size_t irInputNum, size_t irOutputNum)
{
    computeNodeInfo_->node_type_ = opType;
    computeNodeInfo_->node_name_ = opType;
    computeNodeInfo_->ir_inputs_num_ = irInputNum;
    computeNodeInfo_->attr_size_ = 0;
    computeNodeInfo_->ir_outputs_num_ = irOutputNum;
    inputNum_ = 0;
    outputNum_ = 0;
    attrNum_ = 0;
    irInputNum_ = irInputNum;
    irOutputNum_ = irOutputNum;
}

void KernelContextHolder::FinalizeComputeNodeInfo(size_t attrNum)
{
    computeNodeInfo_->inputs_num_ = inputNum_;
    computeNodeInfo_->outputs_num_ = outputNum_;
    attrNum_ = attrNum;
    UpdateAttrDefOffset(inputNum_ + outputNum_, attrNum_);
}

KernelContextHolder::~KernelContextHolder()
{
    FREE(computeNodeInfo_);
    FREE(opInArg_);
}

} // namespace op::internal
