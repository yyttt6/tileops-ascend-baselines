/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_arg.h"
#include "nnopbase_error_msg.h"

namespace op {
namespace internal {
using char_t = char;

void CalcAclTensorNum([[maybe_unused]] size_t idx, const aclTensor* tensor, size_t& num)
{
    if (tensor == nullptr) {
        OP_LOGW("op input is null. idx: %zu.", idx);
        return;
    }
    num++;
}

void CalcAclTensorNum([[maybe_unused]] size_t idx, const aclTensorList* tensor, size_t& num)
{
    if (tensor == nullptr) {
        OP_LOGW("op input tensorlist is null. idx: %zu.", idx);
        return;
    }
    for (size_t i = 0; i < tensor->Size(); i++) {
        CalcAclTensorNum(idx, (*tensor)[i], num);
    }
}

void CalcAclTensorNum(size_t idx, OpArg& arg, size_t& num)
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            CalcAclTensorNum(idx, reinterpret_cast<aclTensor*>(arg->pointer), num);
            break;
        case OpArgType::OPARG_ACLTENSOR_LIST:
            CalcAclTensorNum(idx, reinterpret_cast<aclTensorList*>(arg->pointer), num);
            break;
        default:
            break;
    }
}

const std::string& OpArgTypeStr(int argType)
{
    static const std::array<std::string, static_cast<size_t>(OP_ARG_DEF_BUTT) + 1> OP_ARG_NAME = {
        "input",     /* OP_INPUT_ARG */
        "output",    /* OP_OUTPUT_ARG */
        "attr",      /* OP_ATTR_ARG */
        "workspace", /* OP_WORKSPACE_ARG */
        "outshape",  /* OP_OUTSHAPE_ARG */
        "option",    /* OP_OPTION_ARG */
        "mode",      /* OP_EXEC_MODE_ARG */
        "memset",    /* OP_MEMSET_ARG */
        "empty",     /* OP_EMPTY_ARG */
        "unknown"    /* OP_ARG_DEF_BUTT */
    };

    if (static_cast<size_t>(argType) >= OP_ARG_NAME.size()) {
        return OP_ARG_NAME[OP_ARG_NAME.size() - 1];
    } else {
        return OP_ARG_NAME[argType];
    }
}

aclnnStatus GenKeyByArgImpl(char*& key, const std::array<TensorInfo, MAX_TENSOR_SIZE>& tensorInfos, bool ignoreOptional,
                            size_t idx, const aclTensor* tensor, int argType)
{
    if (tensor == nullptr) {
        OP_LOGW("tensor is nullptr when genKeyByArgImpl.");
        return ACLNN_SUCCESS;
    }

    if (idx >= tensorInfos.size()) {
        OP_LOGE(ACLNN_ERR_INNER, "%s %zu is larger than support nd size %zu.", OpArgTypeStr(argType).c_str(), idx,
                tensorInfos.size());
        return ACLNN_ERR_INNER;
    }

    if (ignoreOptional && tensorInfos[idx].tensorType == TensorType::OPTIONAL) {
        OP_LOGD("GenKey, ignore optional tensor [%zu].", idx);
        return ACLNN_SUCCESS;
    }

    auto dataType = tensor->GetDataType();
    if (tensorInfos[idx].dtMatchMode == DtMatchMode::DTYPE_BYTE) {
        auto typeSize = op::TypeSize(dataType);
        if (typeSize <= MAX_VALID_DTYPE_SIZE) {
            AssignAndIncrement(key, static_cast<char>(typeSize));
            OP_LOGD("GenKey, typeSize is bigger than MAX_VALID_DTYPE_SIZE.");
        } else {
            AssignAndIncrement(key, static_cast<char>(typeSize - kDataTypeSizeBitOffset + MAX_VALID_DTYPE_SIZE));
        }
    } else {
        OP_LOGD("GenKey, tensorInfos[%zu].dtMatchMode is not equal DtMatchMode::DTYPE_BYTE.", idx);
        AssignAndIncrement(key, static_cast<char>(dataType));
    }

    /* 1. If tensor support only ND, it means the operator does not care about the format.
     * We can ignore the tensor format.
     * 2. Otherwise, If tensor support current format, we can use current format for matching.
     * 3. Otherwise, If tensor support ND, it means this tensor can use nd when there are not
     * more accurate formats.
     * 4. Otherwise, just return failed to tell the usr this op cannot use this formats. */
    auto& tensorInfo = tensorInfos[idx];
    if (tensorInfo.fmtInfo.fmtType == FormatType::ONLY_SUPPORT_ND) {
        AssignAndIncrement(key, static_cast<char_t>(ge::FORMAT_ND));
    } else {
        auto primaryFormat = static_cast<ge::Format>(GetPrimaryFormat(tensor->GetStorageFormat()));
        if (tensorInfo.fmtInfo.supportFormats.count(primaryFormat) != 0) {
            AssignAndIncrement(key, primaryFormat);
        } else {
            if (tensorInfo.fmtInfo.fmtType == FormatType::SUPPORT_ND) {
                AssignAndIncrement(key, static_cast<char_t>(ge::FORMAT_ND));
            } else {
                OP_LOGE(ACLNN_ERR_INNER, "Cannot find any bin for %s %zu with [storage format %d, primary format %d].",
                        OpArgTypeStr(argType).c_str(), idx, tensor->GetStorageFormat(),
                        GetPrimaryFormat(tensor->GetStorageFormat()));
                return ACLNN_ERR_INNER;
            }
        }
    }

    GenTransDataKey(key, tensorInfos, idx, tensor);
    GenDimension(key, tensorInfos, idx, tensor);
    return ACLNN_SUCCESS;
}

aclnnStatus GenKeyByArgImpl(char*& key, const std::array<TensorInfo, MAX_TENSOR_SIZE>& tensorInfos, bool ignoreOptional,
                            size_t idx, const aclTensorList* tensorList, int argType)
{
    if (ignoreOptional && idx < tensorInfos.size() && tensorInfos[idx].tensorType == TensorType::OPTIONAL) {
        return ACLNN_SUCCESS;
    }
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGW("tensorList is nullptr or tensorList's size is zero.");
        return ACLNN_ERR_INNER;
    }
    auto data = tensorList->GetData();
    auto ret = GenKeyByArgImpl(key, tensorInfos, ignoreOptional, idx, data[0], argType);
    if (ret != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER;
    }
    auto tensorListSize = tensorInfos[idx].tensorType == TensorType::DYNAMIC_FOLDED ? 1 : tensorList->Size();
    AssignAndIncrement(key, static_cast<char>(tensorListSize + MAX_VALID_DTYPE_FORMAT_KEY));
    return ACLNN_SUCCESS;
}

aclnnStatus GenKeyByArgImpl(char*& key, const std::array<TensorInfo, MAX_TENSOR_SIZE>& tensorInfos, bool ignoreOptional,
                            size_t idx, OpArg& opArg, int argType)
{
    if (opArg.type == OpArgType::OPARG_ACLTENSOR) {
        return GenKeyByArgImpl(key, tensorInfos, ignoreOptional, idx, reinterpret_cast<aclTensor*>(opArg->pointer),
                               argType);
    } else if (opArg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
        return GenKeyByArgImpl(key, tensorInfos, ignoreOptional, idx, reinterpret_cast<aclTensorList*>(opArg->pointer),
                               argType);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GenKeyByOneAttrImplWrapper(char*& key, size_t& remainLen, std::vector<AttrInfo>& attrInfos, OpArg& arg,
                                       size_t idx)
{
    if (attrInfos.size() <= idx) {
        return ACLNN_SUCCESS;
    }
    if (attrInfos[idx].supportAll) {
        return ACLNN_SUCCESS;
    }
    AssignAndIncrement(key, SLASH);
    switch (arg.type) {
        case OpArgType::OPARG_DATATYPE:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], static_cast<op::DataType>(arg->value));
        case OpArgType::OPARG_BOOL:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], static_cast<bool>(arg->value));
        case OpArgType::OPARG_INT:
        case OpArgType::OPARG_UINT:
        case OpArgType::OPARG_IMPLMODE:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], reinterpret_cast<uint64_t>(arg->value));
        case OpArgType::OPARG_FLOAT:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], arg->fvalue);
        case OpArgType::OPARG_DOUBLE:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], arg->dvalue);
        case OpArgType::OPARG_STRING:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], reinterpret_cast<char*>(arg->pointer));
        case OpArgType::OPARG_ACLSCALAR:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx], reinterpret_cast<aclScalar*>(arg->pointer));
        case OpArgType::OPARG_INT_LIST:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx],
                                       reinterpret_cast<aclIntArray*>(arg->pointer));
        case OpArgType::OPARG_FLOAT_LIST:
            return GenKeyByOneAttrImpl(key, remainLen, idx, attrInfos[idx],
                                       reinterpret_cast<aclFloatArray*>(arg->pointer));
        default:
            std::string argTypeRange = "[OPARG_ACLSCALAR(2), OPARG_STRING(3), OPARG_BOOL(4), OPARG_INT(5), "
                                       "OPARG_UINT(6), OPARG_FLOAT(7), OPARG_DOUBLE(8), OPARG_DATATYPE(9), "
                                       "OPARG_INT_LIST(10), OPARG_FLOAT_LIST(12), OPARG_IMPLMODE(14)]";
            auto typeStr = OpArgTypeToStr(arg.type);
            OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(typeStr.GetString(), argTypeRange.c_str());
            return ACLNN_ERR_INNER;
    }
}

thread_local std::vector<LaunchArgInfo::ArgAddr> LaunchArgInfo::addrInfo;

} // namespace internal
} // namespace op