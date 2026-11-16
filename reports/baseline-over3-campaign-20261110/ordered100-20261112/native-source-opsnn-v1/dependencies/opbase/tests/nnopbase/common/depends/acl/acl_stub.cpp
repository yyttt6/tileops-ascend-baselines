/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

size_t aclGetTensorDescNumDims(const aclTensorDesc* desc)
{
    if (desc == nullptr) {
        printf("desc == nullptr \n");
        return 0U;
    }
    if ((desc->dims.size() > 0U) && (desc->dims[0U] == -2)) {
        printf("(desc->dims.size() > 0U) && (desc->dims[0U] == -2) \n");
        return 1U;
    }
    return desc->dims.size();
}

aclError aclGetTensorDescDimV2(const aclTensorDesc* desc, size_t index, int64_t* dimSize)
{
    if (desc == nullptr || dimSize == nullptr || index >= desc->dims.size()) {
        printf("desc == nullptr || dimSize == nullptr || index >= desc->dims.size() \n");
        return ACL_ERROR_INVALID_PARAM;
    }
    *dimSize = desc->dims[index];
    return ACL_SUCCESS;
}

aclTensorDesc::aclTensorDesc(const aclDataType aclTensorDataType, const std::initializer_list<int64_t> shape,
                             const aclFormat aclTensorFormat)
    : dims(shape)
{
    this->dataType = aclTensorDataType;
    this->format = aclTensorFormat;
    this->storageFormat = aclTensorFormat;
    this->storageDims = dims;
}

namespace gert {
GlobalProfilingWrapper::GlobalProfilingWrapper() {}
} // namespace gert