/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>

#include "exe_graph/runtime/tiling_context.h"
#include "kernel_utils.h"
#include "memset_ctx_holder.h"

namespace op::internal {

void MemSetKernelContextHolder::UpdateInputArg(const std::vector<MemSetTensorInfo>& memsetTensorInfo)
{
    for (size_t i = 0; i < memsetTensorInfo.size(); i++) {
        *PtrCastTo<uint64_t>(opInArg_[i + 1].data.inplace) = memsetTensorInfo[i].tensorDataSize_;
    }

    inputNum_ += memsetTensorInfo.size() + 1; // input[0] is not used for workspace checking
}

void MemSetKernelContextHolder::UpdateAttrArg(const std::vector<MemSetTensorInfo>& memsetTensorInfo)
{
    size_t inputNum = memsetTensorInfo.size();
    size_t attrSize = sizeof(gert::TypedContinuousVector<size_t>) + sizeof(size_t) * inputNum;

    // tensor sizes, dtype, int value, float value
    constexpr size_t attrNum = 4;
    std::array<size_t, attrNum> attrSizeArr{attrSize, attrSize, attrSize, attrSize};
    std::array<void*, attrNum> attrAddr{nullptr, nullptr, nullptr, nullptr};

    AppendAttr(attrSizeArr, attrAddr);
    auto attrTensorSize = static_cast<gert::ContinuousVector*>(attrAddr[0]);
    auto attrDtype = static_cast<gert::ContinuousVector*>(attrAddr[1]);
    auto attrValueInt = static_cast<gert::ContinuousVector*>(attrAddr[2]);
    auto attrValueFloat = static_cast<gert::ContinuousVector*>(attrAddr[3]);

    for (auto elem : {attrTensorSize, attrDtype, attrValueInt, attrValueFloat}) {
        elem->Init(inputNum);
        elem->SetSize(inputNum);
    }

    for (size_t i = 0; i < inputNum; i++) {
        const auto& elem = memsetTensorInfo[i];
        size_t* pTensorSize = static_cast<size_t*>(attrTensorSize->MutableData()) + i;
        size_t* pDtype = static_cast<size_t*>(attrDtype->MutableData()) + i;
        size_t* pValueInt = static_cast<size_t*>(attrValueInt->MutableData()) + i;
        size_t* pValueFloat = static_cast<size_t*>(attrValueFloat->MutableData()) + i;

        *pTensorSize = elem.tensorDataSize_;
        *PtrCastTo<op::DataType>(pDtype) = elem.dtype_;
        *PtrCastTo<int64_t>(pValueInt) = elem.valueInt_;
        *PtrCastTo<float32_t>(pValueFloat) = elem.valueFloat_;
    }
}

} // namespace op::internal
