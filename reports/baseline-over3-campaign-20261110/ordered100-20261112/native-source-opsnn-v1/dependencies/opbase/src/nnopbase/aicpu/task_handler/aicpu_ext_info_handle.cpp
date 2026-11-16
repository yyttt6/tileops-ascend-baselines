/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_ext_info_handle.h"
#include <climits>
#include <map>
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "proto/attr_value.pb.h"
#include "proto/fwk_adapter.pb.h"
#include "proto/node_def.pb.h"
#include "acl/acl_rt.h"

namespace op {
namespace internal {
namespace {
using TFDataType = ::domi::tensorflow::DataType;

TFDataType ConvertGeDataType2TfDataType(const ge::DataType dataType, const bool isRef = false)
{
    static const std::map<ge::DataType, TFDataType> dataTypeMap = {
        {ge::DataType::DT_FLOAT16, TFDataType::DT_HALF},
        {ge::DataType::DT_FLOAT, TFDataType::DT_FLOAT},
        {ge::DataType::DT_DOUBLE, TFDataType::DT_DOUBLE},
        {ge::DataType::DT_INT8, TFDataType::DT_INT8},
        {ge::DataType::DT_UINT8, TFDataType::DT_UINT8},
        {ge::DataType::DT_INT16, TFDataType::DT_INT16},
        {ge::DataType::DT_UINT16, TFDataType::DT_UINT16},
        {ge::DataType::DT_INT32, TFDataType::DT_INT32},
        {ge::DataType::DT_UINT32, TFDataType::DT_UINT32},
        {ge::DataType::DT_INT64, TFDataType::DT_INT64},
        {ge::DataType::DT_UINT64, TFDataType::DT_UINT64},
        {ge::DataType::DT_BOOL, TFDataType::DT_BOOL},
        {ge::DataType::DT_RESOURCE, TFDataType::DT_RESOURCE},
        {ge::DataType::DT_STRING, TFDataType::DT_STRING},
        {ge::DataType::DT_STRING_REF, TFDataType::DT_STRING_REF},
        {ge::DataType::DT_COMPLEX64, TFDataType::DT_COMPLEX64},
        {ge::DataType::DT_COMPLEX128, TFDataType::DT_COMPLEX128},
        {ge::DataType::DT_QINT8, TFDataType::DT_QINT8},
        {ge::DataType::DT_QUINT8, TFDataType::DT_QUINT8},
        {ge::DataType::DT_QINT16, TFDataType::DT_QINT16},
        {ge::DataType::DT_QUINT16, TFDataType::DT_QUINT16},
        {ge::DataType::DT_QINT32, TFDataType::DT_QINT32},
        {ge::DataType::DT_VARIANT, TFDataType::DT_VARIANT},
        {ge::DataType::DT_BF16, TFDataType::DT_BFLOAT16},
        {ge::DataType::DT_UINT1, TFDataType::DT_BOOL}};

    const std::map<ge::DataType, TFDataType>::const_iterator& iter = dataTypeMap.find(dataType);
    if (iter != dataTypeMap.end()) {
        if (isRef) {
            const auto diff = TFDataType::DT_FLOAT_REF - TFDataType::DT_FLOAT;
            auto tfDataType = iter->second;
            // Avoid mishandling DT_STRING_REF
            if (tfDataType < diff) {
                tfDataType = static_cast<TFDataType>(tfDataType + diff);
            }
            return tfDataType;
        }
        return iter->second;
    }
    return TFDataType::DT_INVALID;
}

// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = std::numeric_limits<int64_t>::min();
// 为防止和路径3的kernel_id重复，此处的kernel_id从10000开始计数
static std::atomic<std::uint64_t> gKernelId(10000U);
} // namespace

uint64_t AicpuExtInfoHandler::GenerateKernelId() { return gKernelId++; }

aclnnStatus AicpuExtInfoHandler::AppendExtInfoShape(const FVector<const aclTensor*>& tensors,
                                                    aicpu::FWKAdapter::FWKTaskExtInfoType type,
                                                    std::string& taskExtInfo, bool isTf) const
{
    const size_t shapeNum = tensors.size();
    const size_t len = sizeof(AicpuExtInfo) + sizeof(AicpuShapeAndType) * shapeNum;
    std::string s(len, ' ');
    AicpuExtInfo* aicpuExtInfo = reinterpret_cast<AicpuExtInfo*>(s.data());
    aicpuExtInfo->infoType = type;
    aicpuExtInfo->infoLen = sizeof(AicpuShapeAndType) * shapeNum;

    AicpuShapeAndType* inputs = reinterpret_cast<AicpuShapeAndType*>(aicpuExtInfo->infoMsg);
    for (size_t index = 0; index < tensors.size(); ++index) {
        if (tensors[index] == nullptr) {
            continue;
        }
        if (isTf) {
            AICPU_ASSERT_NOTNULL_RETVAL(space_);
            auto space = (AicpuTaskSpace*)space_;
            const bool isRef = space->IsRef(index, type == aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE);
            inputs[index].type = ConvertGeDataType2TfDataType(tensors[index]->GetDataType(), isRef);
        } else {
            inputs[index].type = tensors[index]->GetDataType();
        }
        OP_LOGD("inputs %zu type is %d\n", index, inputs[index].type);
        auto& shape = tensors[index]->GetOriginalShape();
        AICPU_ASSERT_OK_RETVAL(UpdateShape(shape, &inputs[index]));
    }
    taskExtInfo.append(s);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::AppendExtBitMap(std::string& taskExtInfo) const
{
    const size_t len = sizeof(AicpuExtInfo) + sizeof(uint64_t);
    std::string s(len, ' ');
    AicpuExtInfo* aicpuExtInfo = reinterpret_cast<AicpuExtInfo*>(s.data());
    aicpuExtInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
    aicpuExtInfo->infoLen = sizeof(uint64_t);
    uint64_t* bit_map = reinterpret_cast<uint64_t*>(aicpuExtInfo->infoMsg);
    // 0 is for dynamic, 1 is for static
    *bit_map = 0;
    taskExtInfo.append(s);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::AppendSessionInfo(std::string& taskExtInfo) const
{
    const size_t len = sizeof(AicpuExtInfo) + sizeof(AicpuSessionInfo);
    std::string s(len, ' ');
    AicpuExtInfo* aicpuExtInfo = reinterpret_cast<AicpuExtInfo*>(s.data());
    aicpuExtInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
    aicpuExtInfo->infoLen = sizeof(AicpuSessionInfo);
    AicpuSessionInfo* sessionInfo = reinterpret_cast<AicpuSessionInfo*>(aicpuExtInfo->infoMsg);
    sessionInfo->sessionId = 0;
    sessionInfo->kernelId = GenerateKernelId();
    sessionInfo->sessFlag = true;
    taskExtInfo.append(s);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::AppendExtOpName(std::string& taskExtInfo) const
{
    const size_t len = sizeof(AicpuExtInfo) + nodeName_.size();
    std::string s(len, ' ');
    AicpuExtInfo* aicpuExtInfo = reinterpret_cast<AicpuExtInfo*>(s.data());
    aicpuExtInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_OP_NAME;
    aicpuExtInfo->infoLen = nodeName_.size();
    char* opName_buf = reinterpret_cast<char*>(aicpuExtInfo->infoMsg);
    if (strcpy_s(opName_buf, nodeName_.size() + 1, nodeName_.c_str()) != EOK) {
        OP_LOGE(ACLNN_ERR_INNER, "AppendExtOpName[%s] Failed to copy opName", nodeName_.c_str());
        return ACLNN_ERR_INNER;
    }

    taskExtInfo.append(s);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::AppendExtShapeType(std::string& taskExtInfo) const
{
    const size_t len = sizeof(AicpuExtInfo) + sizeof(int32_t);
    std::string s(len, ' ');
    AicpuExtInfo* aicpuExtInfo = reinterpret_cast<AicpuExtInfo*>(s.data());
    aicpuExtInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    aicpuExtInfo->infoLen = sizeof(int32_t);
    int32_t* shape_type = reinterpret_cast<int32_t*>(aicpuExtInfo->infoMsg);
    *shape_type = unknownType_;
    taskExtInfo.append(s);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::GenTfExtBuffer(const FVector<const aclTensor*>& inputs,
                                                const FVector<aclTensor*>& outputs, std::string& taskExtInfo) const
{
    // WARNING: OP NAME MUST BE THE FIRST EXTEND INFO FOR RUNTIME!!!
    AICPU_ASSERT_OK_RETVAL(AppendExtOpName(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtShapeType(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtBitMap(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtInfoShape(inputs, aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE, taskExtInfo, true));
    FVector<const aclTensor*> tmp_outputs;
    tmp_outputs.insert(tmp_outputs.cend(), outputs.cbegin(), outputs.cend());
    AICPU_ASSERT_OK_RETVAL(
        AppendExtInfoShape(tmp_outputs, aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE, taskExtInfo, true));
    // tf no need session info in ext info
    return OK;
}

aclnnStatus AicpuExtInfoHandler::GenCCExtBuffer(const FVector<const aclTensor*>& inputs,
                                                const FVector<aclTensor*>& outputs, std::string& taskExtInfo) const
{
    // WARNING: OP NAME MUST BE THE FIRST EXTEND INFO FOR RUNTIME!!!
    AICPU_ASSERT_OK_RETVAL(AppendExtOpName(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtShapeType(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtBitMap(taskExtInfo));
    AICPU_ASSERT_OK_RETVAL(AppendExtInfoShape(inputs, aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE, taskExtInfo));
    FVector<const aclTensor*> tmp_outputs;
    tmp_outputs.insert(tmp_outputs.cend(), outputs.cbegin(), outputs.cend());
    AICPU_ASSERT_OK_RETVAL(AppendExtInfoShape(tmp_outputs, aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE, taskExtInfo));
    // session info for 自研算子
    AICPU_ASSERT_OK_RETVAL(AppendSessionInfo(taskExtInfo));
    return OK;
}

aclnnStatus AicpuExtInfoHandler::Parse(const std::string& extInfo, uint8_t* hostAddr)
{
    AICPU_ASSERT_TRUE_RETVAL(!extInfo.empty());
    extInfoLen_ = extInfo.size();
    extInfo_ = hostAddr;

    if (memcpy_s(extInfo_, extInfoLen_, extInfo.c_str(), extInfoLen_) != EOK) {
        OP_LOGE(ACLNN_ERR_INNER, "[Update][extInfo_][%s] Failed to copy ext info", nodeName_.c_str());
        return ACLNN_ERR_INNER;
    }

    inputShapeAndType_.clear();
    outputShapeAndType_.clear();

    const auto extInfoData = extInfo_;
    size_t offset = 0UL;
    while ((offset + sizeof(AicpuExtInfo)) <= extInfoLen_) {
        auto tmp_extInfoData = PtrAdd(extInfoData, extInfoLen_, offset);
        AICPU_ASSERT_NOTNULL_RETVAL(tmp_extInfoData);
        auto& aicpuExtInfo = *(PtrToPtr<uint8_t, AicpuExtInfo>(tmp_extInfoData));
        OP_LOGD("Ext infoType=%d, infoLen=%u.", aicpuExtInfo.infoType, aicpuExtInfo.infoLen);
        switch (aicpuExtInfo.infoType) {
            case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
                AICPU_ASSERT_OK_RETVAL(ParseExtInputShape(aicpuExtInfo));
                break;
            case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
                outputShapeOffset_ = offset + sizeof(AicpuExtInfo);
                outputShapeLen_ = aicpuExtInfo.infoLen;
                outputShape_.resize(outputNum_);
                AICPU_ASSERT_OK_RETVAL(ParseExtOutputShape(aicpuExtInfo));
                break;
            default:
                OP_LOGD("Node[%s] ignore infoType=%d, infoLen=%u.", nodeName_.c_str(), aicpuExtInfo.infoType,
                        aicpuExtInfo.infoLen);
                break;
        }
        offset += sizeof(AicpuExtInfo);
        offset += aicpuExtInfo.infoLen;
    }

    AICPU_ASSERT_TRUE_RETVAL(offset == extInfoLen_);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo& aicpuExtInfo)
{
    AICPU_ASSERT_TRUE_RETVAL(aicpuExtInfo.infoLen == (inputNum_ * sizeof(AicpuShapeAndType)));
    const auto input = PtrToPtr<char, AicpuShapeAndType>(aicpuExtInfo.infoMsg);

    for (uint32_t index = 0U; index < inputNum_; ++index) {
        inputShapeAndType_.emplace_back(
            PtrAdd<AicpuShapeAndType>(input, static_cast<size_t>(inputNum_), static_cast<size_t>(index)));
    }
    OP_LOGD("Node[%s] parse ext input shape success infoLen=%u.", nodeName_.c_str(), aicpuExtInfo.infoLen);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo& aicpuExtInfo)
{
    AICPU_ASSERT_TRUE_RETVAL(aicpuExtInfo.infoLen == (outputNum_ * sizeof(AicpuShapeAndType)));
    const auto output = PtrToPtr<char, AicpuShapeAndType>(aicpuExtInfo.infoMsg);

    for (uint32_t index = 0U; index < outputNum_; ++index) {
        outputShapeAndType_.emplace_back(
            PtrAdd<AicpuShapeAndType>(output, static_cast<size_t>(outputNum_), static_cast<size_t>(index)));
    }
    OP_LOGD("Node[%s] parse ext output shape success infoLen=%u.", nodeName_.c_str(), aicpuExtInfo.infoLen);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::UpdateInputShape(const uint32_t inputIndex, const gert::Shape& inputShape)
{
    AICPU_ASSERT_TRUE_RETVAL(inputIndex < inputNum_);
    OP_LOGI("input[%u] shape is %s", inputIndex, op::ToString(inputShape).GetString());
    return UpdateShape(inputShape, inputShapeAndType_[static_cast<size_t>(inputIndex)]);
}

aclnnStatus AicpuExtInfoHandler::UpdateOutputShape(const uint32_t outputIndex, const gert::Shape& outputShape)
{
    AICPU_ASSERT_TRUE_RETVAL(outputIndex < outputNum_);
    OP_LOGI("output[%u] shape is %s", outputIndex, op::ToString(outputShape).GetString());
    return UpdateShape(outputShape, outputShapeAndType_[static_cast<size_t>(outputIndex)]);
}

aclnnStatus AicpuExtInfoHandler::UpdateShape(const gert::Shape& shape, AicpuShapeAndType* const shapeAndType)
{
    const auto dimNum = shape.GetDimNum();
    if (dimNum > 8) {
        OP_LOGE(ACLNN_ERR_INNER, "shape dim num should be less than 8, but got [%zu].", dimNum);
        return ACLNN_ERR_INNER;
    }
    size_t index = 0U;
    for (; index < dimNum; ++index) {
        shapeAndType->dims[index] = shape.GetDim(index);
    }
    if (index < aicpu::FWKAdapter::kMaxShapeDims) {
        shapeAndType->dims[index] = kDimEndFlag;
    }

    return OK;
}

aclnnStatus AicpuExtInfoHandler::GetExtInfoDeviceBuffer(const aclOpExecutor* executor, const uint64_t deviceExtMemSize,
                                                        uint64_t& deviceCacheOffset)
{
    if (deviceExtMemSize > executor->workspaceDeviceAicpuMem_) {
        OP_LOGE(ACLNN_ERR_INNER, "deviceExtMemSize:%lu is too big, workspaceDeviceAicpuMem_:%lu", deviceExtMemSize,
                executor->workspaceDeviceAicpuMem_);
        return ACLNN_ERR_INNER;
    }

    deviceExtInfo_ = ValueToPtr(PtrToValue(executor->GetWorkspaceAddr()) + executor->workspaceDeviceAicpuTaskOffset_);
    deviceCacheOffset += deviceExtMemSize;
    OP_LOGI("alloc deviceExtInfo success. deviceExtMemSize:%lu", deviceExtMemSize);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::CopyH2D(const aclrtStream stream, const aclOpExecutor* executor,
                                         const uint64_t deviceExtMemSize, uint64_t& deviceCacheOffset)
{
    if ((unknownType_ == ge::DEPEND_SHAPE_RANGE) && (extInfoLen_ > 0)) {
        AICPU_ASSERT_NOTNULL_RETVAL(extInfo_);
        if (GetExtInfoDeviceBuffer(executor, deviceExtMemSize, deviceCacheOffset) != OK) {
            return ACLNN_ERR_INNER;
        }
        if (deviceExtInfo_ != nullptr) {
            AICPU_ASSERT_RTOK_RETVAL(aclrtMemcpyAsync(deviceExtInfo_, extInfoLen_, extInfo_, extInfoLen_,
                                                      ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
        }
    }
    return OK;
}

aclnnStatus AicpuExtInfoHandler::CopyOutputShapeD2H()
{
    if (outputShapeLen_ > 0) {
        AICPU_ASSERT_NOTNULL_RETVAL(deviceExtInfo_);
        auto outputShapeDeviceAddr = ValueToPtr(PtrToValue(deviceExtInfo_) + outputShapeOffset_);
        const size_t hostLen = outputShape_.size() * sizeof(AicpuShapeAndType);
        AICPU_ASSERT_RTOK_RETVAL(aclrtMemcpy(outputShape_.data(), hostLen, outputShapeDeviceAddr, outputShapeLen_,
                                             ACL_MEMCPY_DEVICE_TO_HOST));
    }
    return OK;
}

aclnnStatus AicpuExtInfoHandler::GetOutputShapeAndType(const uint32_t outputIndex, gert::Shape& shape,
                                                       ge::DataType& dataType) const
{
    AICPU_ASSERT_TRUE_RETVAL(outputIndex < outputNum_);
    GetShapeAndType(outputShape_[static_cast<size_t>(outputIndex)], shape, dataType);
    return OK;
}

void AicpuExtInfoHandler::GetShapeAndType(const AicpuShapeAndType& shapeAndType, gert::Shape& shape,
                                          ge::DataType& dataType)
{
    shape.SetDimNum(0U);
    for (uint32_t i = 0U; i < aicpu::FWKAdapter::kMaxShapeDims; ++i) {
        if (shapeAndType.dims[i] == kDimEndFlag) {
            break;
        }
        shape.AppendDim(shapeAndType.dims[i]);
    }
    dataType = static_cast<ge::DataType>(shapeAndType.type);
}

aclnnStatus AicpuExtInfoHandler::UpdateOutputShapeFromExtInfo(const FVector<aclTensor*>& outputs, aclrtStream stream)
{
    AICPU_ASSERT_RTOK_RETVAL(aclrtSynchronizeStream(stream));
    AICPU_ASSERT_OK_RETVAL(CopyOutputShapeD2H());
    RecordAicpuTime(kShapeD2hCopyEnd);
    for (size_t i = 0U; i < outputs.size(); i++) {
        ge::DataType type;
        AICPU_ASSERT_OK_RETVAL(GetOutputShapeAndType(i, const_cast<gert::Shape&>(outputs[i]->GetViewShape()), type));
        AICPU_ASSERT_OK_RETVAL(GetOutputShapeAndType(i, const_cast<gert::Shape&>(outputs[i]->GetStorageShape()), type));
        AICPU_ASSERT_OK_RETVAL(
            GetOutputShapeAndType(i, const_cast<gert::Shape&>(outputs[i]->GetOriginalShape()), type));
        OP_LOGI("output[%zu], ViewShape is %s, StorageShape is %s, OriginalShape is %s.", i,
                op::ToString(outputs[i]->GetViewShape()).GetString(),
                op::ToString(outputs[i]->GetStorageShape()).GetString(),
                op::ToString(outputs[i]->GetOriginalShape()).GetString());
    }
    return OK;
}

aclnnStatus AicpuExtInfoHandler::UpdateInputAndOutputShape(const FVector<const aclTensor*>& inputs,
                                                           const FVector<aclTensor*>& outputs, aclrtStream stream,
                                                           const aclOpExecutor* executor,
                                                           const uint64_t deviceExtMemSize, uint64_t& deviceCacheOffset)
{
    for (size_t i = 0U; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        AICPU_ASSERT_OK_RETVAL(UpdateInputShape(i, inputs[i]->GetOriginalShape()));
    }
    for (size_t i = 0U; i < outputs.size(); ++i) {
        AICPU_ASSERT_OK_RETVAL(UpdateOutputShape(i, outputs[i]->GetOriginalShape()));
    }
    deviceCacheOffset = executor->workspaceDeviceAicpuTaskOffset_;
    RecordAicpuTime(kUpdateShapeEnd);
    AICPU_ASSERT_OK_RETVAL(CopyH2D(stream, executor, deviceExtMemSize, deviceCacheOffset));
    RecordAicpuTime(kShapeH2DEnd);
    return OK;
}

aclnnStatus AicpuExtInfoHandler::UpdateKernelId()
{
    const auto extInfoData = extInfo_;
    size_t offset = 0UL;
    while ((offset + sizeof(AicpuExtInfo)) <= extInfoLen_) {
        auto tmp_extInfoData = PtrAdd(extInfoData, extInfoLen_, offset);
        AICPU_ASSERT_NOTNULL_RETVAL(tmp_extInfoData);
        auto& aicpuExtInfo = *(PtrToPtr<uint8_t, AicpuExtInfo>(tmp_extInfoData));
        OP_LOGD("Ext infoType=%d, infoLen=%u.", aicpuExtInfo.infoType, aicpuExtInfo.infoLen);
        if (aicpuExtInfo.infoType == aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO) {
            AICPU_ASSERT_TRUE_RETVAL(aicpuExtInfo.infoLen == sizeof(AicpuSessionInfo));
            AicpuSessionInfo* sessionInfo = reinterpret_cast<AicpuSessionInfo*>(aicpuExtInfo.infoMsg);
            sessionInfo->kernelId = GenerateKernelId();
            OP_LOGI("Node[%s] update kernelid=%lu.", nodeName_.c_str(), sessionInfo->kernelId);
            break;
        }
        offset += sizeof(AicpuExtInfo);
        offset += aicpuExtInfo.infoLen;
    }
    return OK;
}
} // namespace internal
} // namespace op
