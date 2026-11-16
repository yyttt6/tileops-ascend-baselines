/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shape_inference.h"

#include "exe_graph/runtime/tensor.h"
#include "transfer_shape_according_to_format_ext.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"

namespace op::internal {

const size_t EXT_INDEX_INPUT_SIZE = 0;
const size_t EXT_INDEX_HIDDEN_SIZE = 1;
const size_t EXT_INDEX_STATE_SIZE = 2;

aclnnStatus UpdateOutputShape(aclTensor* tensor)
{
    auto& originalShape = tensor->GetOriginalShape();
    tensor->SetStorageShape(originalShape);
    transformer::ExtAxisOpValue extAxisOpValue;
    extAxisOpValue[EXT_INDEX_INPUT_SIZE] = 1;
    extAxisOpValue[EXT_INDEX_HIDDEN_SIZE] = 1;
    extAxisOpValue[EXT_INDEX_STATE_SIZE] = -1;
    auto isSuccess = transformer::ShapeTransferAccordingToFormatExt::TransferShape(
        tensor->GetOriginalFormat(), tensor->GetStorageFormat(), tensor->GetDataType(),
        tensor->GetTensor()->MutableStorageShape(), extAxisOpValue, GetCurrentPlatformInfo().GetPlatformInfos());
    if (!isSuccess) {
        return ACLNN_ERR_INNER;
    }
    tensor->SetViewShape(originalShape);
    return ACLNN_SUCCESS;
}

aclnnStatus UpdateOutputShape(aclTensorList* tensors)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        CHECK_RET_CODE(UpdateOutputShape((*tensors)[i]), "update dynamic output error.");
    }
    return ACLNN_SUCCESS;
}
} // end namespace op::internal
