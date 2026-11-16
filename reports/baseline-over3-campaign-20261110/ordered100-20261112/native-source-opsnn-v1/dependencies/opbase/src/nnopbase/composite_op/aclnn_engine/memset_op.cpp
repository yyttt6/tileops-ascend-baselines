/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memset_op.h"

#include "aclnn/acl_meta.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "kernel_arg.h"
#include "../../common/inc/kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "op_kernel.h"
#include "bridge_pool.h"

namespace op {
namespace internal {

MemsetV2ArgContext::MemsetV2ArgContext()
{
    hugeMemPoolIndex_ = GetPoolIndex();
    if (hugeMemPoolIndex_ != op::kInvalidHugeMemIndexId) {
        UpdateHugeMemIndex(op::kInvalidHugeMemIndexId);
    }
}

void MemsetV2ArgContext::AddOneMemSetTensor(const aclTensor* tensor, op::DataType dtype, int64_t valueInt,
                                            float valueFloat)
{
    memsetInputs_.emplace_back(tensor);
    if (dtype == op::DataType::DT_INT32 || dtype == op::DataType::DT_UINT32) {
        memsetIntAttrs_.emplace_back(valueInt);
    } else {
        memsetFloatAttrs_.emplace_back(valueFloat);
    }
}

aclnnStatus MemsetV2ArgContext::Init(const std::vector<MemSetTensorInfo>& memsetTensorInfo)
{
    for (const auto& elem : memsetTensorInfo) {
        if (elem.argType_ == OpArgType::OPARG_ACLTENSOR) {
            if (elem.tensor_ == nullptr) {
                OP_LOGW("elem idx [%zu] tensor is nullptr.", elem.argIdx_);
                continue;
            }
            AddOneMemSetTensor(elem.tensor_, elem.dtype_, elem.valueInt_, elem.valueFloat_);
        } else if (elem.argType_ == OpArgType::OPARG_ACLTENSOR_LIST) {
            if (elem.tensorList_ == nullptr) {
                OP_LOGW("elem idx [%zu] tensor list is nullptr.", elem.argIdx_);
                continue;
            }
            for (size_t i = 0; i < elem.tensorList_->Size(); i++) {
                const aclTensor* tensor = (*elem.tensorList_)[i];
                if (tensor == nullptr) {
                    OP_LOGW("elem idx [%zu] tensorlist[%zu] is nullptr.", elem.argIdx_, i);
                    continue;
                }
                AddOneMemSetTensor(tensor, elem.dtype_, elem.valueInt_, elem.valueFloat_);
            }
        }
    }
    OP_CHECK(!memsetInputs_.empty(), OP_LOGW("no arg need to memset"), return ACLNN_SUCCESS);

    aclOpExecutor* executor = op::internal::GetThreadLocalContext().executor_;
    CHECK_COND(executor != nullptr, ACLNN_ERR_INNER_NULLPTR, "executor is nullptr");
    memsetTensors_ = executor->AllocTensorList(memsetInputs_.data(), memsetInputs_.size());
    intAttrArray_ = executor->AllocIntArray(memsetIntAttrs_.data(), memsetIntAttrs_.size());
    floatAttrArray_ = executor->AllocFloatArray(memsetFloatAttrs_.data(), memsetFloatAttrs_.size());
    CHECK_COND((memsetTensors_ != nullptr && intAttrArray_ != nullptr && floatAttrArray_ != nullptr),
               ACLNN_ERR_INNER_NULLPTR, "Create memsetv2 args failed");

    const aclTensor* memsetV2WsTensor = memsetInputs_[0];
    memsetV2OpArgCtx_ = GetOpArgContext(OP_INPUT(memsetTensors_), OP_OUTPUT(memsetTensors_),
                                        OP_WORKSPACE(memsetV2WsTensor), OP_ATTR(intAttrArray_, floatAttrArray_));
    CHECK_COND(memsetV2OpArgCtx_ != nullptr, ACLNN_ERR_INNER_NULLPTR, "Create memsetv2 arg ctx failed");
    return ACLNN_SUCCESS;
}

MemsetV2ArgContext::~MemsetV2ArgContext()
{
    op::DestroyOpArgContext(memsetV2OpArgCtx_);

    if (hugeMemPoolIndex_ != op::kInvalidHugeMemIndexId) {
        UpdateHugeMemIndex(hugeMemPoolIndex_);
    }
}

OpArgContext* MemsetV2ArgContext::GetMemsetV2OpArgContext() { return memsetV2OpArgCtx_; }

class MemSetOpKernel : public OpKernel {
public:
    OpKernelBin* SelectBin(MemsetVersion memsetVersion, size_t num)
    {
        if (bins_.size() == 0) {
            return nullptr;
        }

        if (memsetVersion == MemsetVersion::MEMSET_V1_ASCENDC || memsetVersion == MemsetVersion::MEMSET_V2) {
            OP_CHECK(bins_.begin()->second->JsonLoad() == ACLNN_SUCCESS,
                     OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "json load failed."), return nullptr);
            return bins_.begin()->second.get();
        }

        for (auto& elem : bins_) {
            elem.second->JsonLoad();
            auto& opJson = elem.second->binJson_.GetVar();
            if (!opJson.contains("supportInfo") || !(opJson["supportInfo"].contains("attrs")) ||
                opJson["supportInfo"]["attrs"].size() < 1 || !opJson["supportInfo"]["attrs"][0].contains("value")) {
                OP_LOGW("json parse error. does not contain supportInfo or attrs.");
                continue;
            }
            if (opJson["supportInfo"]["attrs"][0]["value"].size() == num) {
                return elem.second.get();
            }
        }
        OP_LOGE(ACLNN_ERR_INNER, "Memset bin not found. num [%zu]", num);
        return nullptr;
    }
};

static inline uint32_t GetMemsetOpId(MemsetVersion memsetVersion)
{
    if (memsetVersion == MemsetVersion::MEMSET_V2) {
        return OpTypeDict::ToOpType(MEMSET_V2_NAME);
    }
    return OpTypeDict::ToOpType(MEMSET_V1_NAME);
}

aclnnStatus SelectMemsetOpBin(MemsetVersion memsetVersion, size_t inputNum, OpKernelBin*& opBin)
{
#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
    uint32_t opid = GetMemsetOpId(memsetVersion);
#else
    static uint32_t opid = GetMemsetOpId(memsetVersion);
#endif
    MemSetOpKernel* kernel = static_cast<MemSetOpKernel*>(gKernelMgr.GetKernel(opid));
    if (kernel == nullptr) {
        return ACLNN_ERR_INNER;
    }

    (void)gKernelMgr.AclOpKernelInit(opid);
    opBin = kernel->SelectBin(memsetVersion, inputNum);
    if (!opBin) {
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

} // namespace internal
} // namespace op
