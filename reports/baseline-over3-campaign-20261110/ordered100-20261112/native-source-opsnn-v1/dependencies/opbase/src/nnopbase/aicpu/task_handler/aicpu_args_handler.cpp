/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_args_handler.h"
#include "opdev/aicpu/aicpu_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aicpu_task_struct.h"
#include "exe_graph/runtime/tensor.h"

#include "utils/extern_math_util.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "proto/attr_value.pb.h"
#include "proto/cpu_proto/cpu_attr.pb.h"
#include "proto/cpu_proto/cpu_node_def.pb.h"
#include "proto/cpu_proto/cpu_tensor.pb.h"
#include "proto/fwk_adapter.pb.h"
#include "proto/node_def.pb.h"

using namespace ge;

namespace op {
namespace internal {
namespace {
constexpr uint16_t kModelId = 36; // AICPU
constexpr size_t kMaxTotalHostLen = 1024U;

using TFDataType = ::domi::tensorflow::DataType;
using namespace std;
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

aclnnStatus GetAttrValueFromGe(const GeAttrValue& geAttrValue, domi::tensorflow::AttrValue& attrValue)
{
    const GeAttrValue::ValueType geValueType = geAttrValue.GetValueType();
    switch (geValueType) {
        case GeAttrValue::ValueType::VT_STRING: {
            std::string stringValue;
            geAttrValue.GetValue<std::string>(stringValue);
            attrValue.set_s(stringValue);
            break;
        }
        case GeAttrValue::ValueType::VT_FLOAT: {
            float floatValue = 0.0;
            geAttrValue.GetValue<float>(floatValue);
            attrValue.set_f(floatValue);
            break;
        }
        case GeAttrValue::ValueType::VT_BOOL: {
            bool boolValue = false;
            geAttrValue.GetValue<bool>(boolValue);
            attrValue.set_b(boolValue);
            break;
        }
        case GeAttrValue::ValueType::VT_INT: {
            int64_t intValue = 0;
            geAttrValue.GetValue<int64_t>(intValue);
            attrValue.set_i(intValue);
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_FLOAT: {
            std::vector<float> floatList;
            geAttrValue.GetValue<std::vector<float>>(floatList);
            AICPU_ASSERT_NOTNULL_RETVAL(attrValue.mutable_list());
            for (const float& float_element : floatList) {
                attrValue.mutable_list()->add_f(float_element);
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_LIST_INT: {
            std::vector<std::vector<int64_t>> shapeValue;
            geAttrValue.GetValue<std::vector<std::vector<int64_t>>>(shapeValue);
            auto attrList = attrValue.mutable_list();
            AICPU_ASSERT_NOTNULL_RETVAL(attrList);
            for (const std::vector<int64_t>& shape : shapeValue) {
                domi::tensorflow::TensorShapeProto* shape_tensor = attrList->add_shape();
                AICPU_ASSERT_NOTNULL_RETVAL(shape_tensor);
                for (const int64_t& dim : shape) {
                    if (dim <= 0) {
                        shape_tensor->set_unknown_rank(true);
                        // If unknown_rank true, dim.size() must be 0.
                        shape_tensor->clear_dim();
                        break;
                    }
                    shape_tensor->add_dim()->set_size(dim);
                }
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_DATA_TYPE: {
            std::vector<ge::DataType> geTypeList;
            geAttrValue.GetValue<std::vector<ge::DataType>>(geTypeList);
            AICPU_ASSERT_NOTNULL_RETVAL(attrValue.mutable_list());
            for (ge::DataType& geType : geTypeList) {
                domi::tensorflow::DataType dataType = ConvertGeDataType2TfDataType(geType);
                attrValue.mutable_list()->add_type(dataType);
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_INT: {
            std::vector<int64_t> listIntValue;
            geAttrValue.GetValue<std::vector<int64_t>>(listIntValue);
            AICPU_ASSERT_NOTNULL_RETVAL(attrValue.mutable_list());
            for (int32_t value : listIntValue) {
                attrValue.mutable_list()->add_i(value);
            }
            break;
        }
        case GeAttrValue::ValueType::VT_DATA_TYPE: {
            ge::DataType geType = DT_MAX;
            geAttrValue.GetValue<ge::DataType>(geType);
            domi::tensorflow::DataType dataType = ConvertGeDataType2TfDataType(geType);
            attrValue.set_type(dataType);
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_STRING: {
            std::vector<std::string> listStringValue;
            geAttrValue.GetValue<std::vector<std::string>>(listStringValue);
            AICPU_ASSERT_NOTNULL_RETVAL(attrValue.mutable_list());
            for (std::string value : listStringValue) {
                attrValue.mutable_list()->add_s(value);
            }
            break;
        }
        default: { // todo some other attr type
            OP_LOGE(ACLNN_ERR_INNER, "Currently ir2tf can not support attr type[%d].", geValueType);
            return ACLNN_ERR_INNER;
        }
    }
    return OK;
}

using AttrValueMap = google::protobuf::Map<std::string, domi::tensorflow::AttrValue>;

aclnnStatus ParseAttr(const FVector<const aclTensor*>& inputs, const AicpuAttrs& attrs, const std::string& opType,
                      domi::tensorflow::NodeDef& nodeDef)
{
    nodeDef.set_op(opType);

    for (auto iter = attrs.begin(); iter != attrs.end(); ++iter) {
        domi::tensorflow::AttrValue attrValue;
        AICPU_ASSERT_OK_RETVAL(GetAttrValueFromGe(iter->second, attrValue));

        const std::string& geAttrName = iter->first;
        AICPU_ASSERT_NOTNULL_RETVAL(nodeDef.mutable_attr());
        auto pair = nodeDef.mutable_attr()->insert(AttrValueMap::value_type(geAttrName, attrValue));
        AICPU_ASSERT_TRUE_RETVAL(pair.second);
    }

    if (!inputs.empty() && inputs[0] != nullptr) {
        domi::tensorflow::AttrValue attrValue;
        auto geDtype = inputs[0]->GetDataType();
        auto tfDtype = ConvertGeDataType2TfDataType(geDtype);
        attrValue.set_type(tfDtype);
        // 外部可能传入 "T" 属性，此时会插入失败，不做校验
        (void)nodeDef.mutable_attr()->insert(AttrValueMap::value_type("T", attrValue));
    }
    return OK;
}

aclnnStatus SerializeNodedefToBuffer(const domi::tensorflow::NodeDef& nodeDef, uint8_t* buffer, size_t nodeDefSize)
{
    google::protobuf::io::ArrayOutputStream arrayStream(buffer, static_cast<int32_t>(nodeDefSize));
    google::protobuf::io::CodedOutputStream outputStream(&arrayStream);
    outputStream.SetSerializationDeterministic(true);
    AICPU_ASSERT_TRUE_RETVAL(nodeDef.SerializeToCodedStream(&outputStream));
    return OK;
}

// for KernelRunParam
aclnnStatus SetTensorDataInfo(const aclTensor& aclTensor, aicpu::FWKAdapter::TensorDataInfo& tensorDataInfo,
                              const bool isRef)
{
    const auto dataType = aclTensor.GetDataType();
    uint32_t tfDataType = static_cast<uint32_t>(ConvertGeDataType2TfDataType(dataType, isRef));
    tensorDataInfo.set_dtype(tfDataType);
    tensorDataInfo.set_data_addr(ULLONG_MAX);

    auto& shape = aclTensor.GetOriginalShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        AICPU_ASSERT_TRUE_RETVAL(shape.GetDim(i) >= 0);
        tensorDataInfo.add_dim(shape.GetDim(i));
    }
    return OK;
}

aclnnStatus BuildKernelRunParam(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                                const AicpuTaskSpace* space, aicpu::FWKAdapter::KernelRunParam& kernelRunParam)
{
    AICPU_ASSERT_NOTNULL_RETVAL(space);
    for (size_t i = 0; i < inputs.size(); i++) {
        auto inputTensor = kernelRunParam.add_input();
        const auto aclTensor = inputs[i];
        if (aclTensor == nullptr) {
            continue;
        }
        AICPU_ASSERT_NOTNULL_RETVAL(inputTensor);
        AICPU_ASSERT_NOTNULL_RETVAL(aclTensor);
        const bool isRef = space->IsRef(i, true);
        AICPU_ASSERT_OK_RETVAL(SetTensorDataInfo(*aclTensor, *inputTensor, isRef));
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto outputTensor = kernelRunParam.add_output();
        const auto aclTensor = outputs[i];
        AICPU_ASSERT_NOTNULL_RETVAL(outputTensor);
        AICPU_ASSERT_NOTNULL_RETVAL(aclTensor);
        const bool isRef = space->IsRef(i, false);
        AICPU_ASSERT_OK_RETVAL(SetTensorDataInfo(*aclTensor, *outputTensor, isRef));
    }
    return OK;
}

aclnnStatus SerializeKernelRunParamToBuffer(const aicpu::FWKAdapter::KernelRunParam& kernelRunParam, uint8_t* buffer)
{
    google::protobuf::io::ArrayOutputStream arrayStream(buffer, static_cast<int32_t>(kernelRunParam.ByteSizeLong()));
    google::protobuf::io::CodedOutputStream outputStream(&arrayStream);
    outputStream.SetSerializationDeterministic(true);
    AICPU_ASSERT_TRUE_RETVAL(kernelRunParam.SerializeToCodedStream(&outputStream));
    return OK;
}

uint64_t RoundUp(const uint64_t originValue, const uint64_t multipleOf)
{
    if (multipleOf == 0) {
        return 0;
    }
    return (originValue + multipleOf - 1) / multipleOf * multipleOf;
}

aclnnStatus UpdateProtoTensor(const aclTensor* aclTensor, aicpuops::Tensor* tensor)
{
    if (aclTensor == nullptr) {
        return OK;
    }
    auto aicpuShape = tensor->mutable_tensor_shape();
    AICPU_ASSERT_NOTNULL_RETVAL(aicpuShape);
    auto shape = aclTensor->GetOriginalShape();
    const auto dimNum = shape.GetDimNum();
    for (size_t index = 0U; index < dimNum; ++index) {
        aicpuops::TensorShape_Dim* aicpuDims = aicpuShape->add_dim();
        AICPU_ASSERT_NOTNULL_RETVAL(aicpuDims);
        aicpuDims->set_size(shape.GetDim(index));
    }
    aicpuShape->set_data_format(aclTensor->GetOriginalFormat());
    aicpuShape->set_unknown_rank(true); // 默认动态
    tensor->set_tensor_type(aclTensor->GetDataType());
    OP_LOGD("aclTensor format is %s, dtype is %s", op::ToString(aclTensor->GetOriginalFormat()).GetString(),
            op::ToString(aclTensor->GetDataType()).GetString());
    return OK;
}

aclnnStatus GetAttrValueFromGeAttrValue(const std::string& opType, const GeAttrValue& geAttrValue,
                                        aicpuops::AttrValue& attrValue)
{
    const GeAttrValue::ValueType geValueType = geAttrValue.GetValueType();
    switch (geValueType) {
        case GeAttrValue::ValueType::VT_STRING: {
            std::string stringValue;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::string>(stringValue));
            attrValue.set_s(stringValue);
            break;
        }
        case GeAttrValue::ValueType::VT_FLOAT: {
            float floatValue = 0.0;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<float>(floatValue));
            attrValue.set_f(floatValue);
            break;
        }
        case GeAttrValue::ValueType::VT_BOOL: {
            bool boolValue = false;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<bool>(boolValue));
            attrValue.set_b(boolValue);
            break;
        }
        case GeAttrValue::ValueType::VT_INT: {
            int64_t intValue = 0;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<int64_t>(intValue));
            attrValue.set_i(intValue);
            break;
        }
        case GeAttrValue::ValueType::VT_DATA_TYPE: {
            ge::DataType geType = DT_MAX;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<ge::DataType>(geType));
            attrValue.set_type(geType);
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_FLOAT: {
            std::vector<float> floatList;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::vector<float>>(floatList));
            auto array = attrValue.mutable_array();
            AICPU_ASSERT_NOTNULL_RETVAL(array);
            for (const float& f : floatList) {
                array->add_f(f);
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_LIST_INT: {
            std::vector<std::vector<int64_t>> shapeValue;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::vector<std::vector<int64_t>>>(shapeValue));
            auto attrList = attrValue.mutable_array();
            AICPU_ASSERT_NOTNULL_RETVAL(attrList);
            for (const vector<int64_t>& shape : shapeValue) {
                aicpuops::TensorShape* shape_tensor = attrList->add_shape();
                AICPU_ASSERT_NOTNULL_RETVAL(shape_tensor);
                for (const int64_t& dim : shape) {
                    if (dim < 0) {
                        OP_LOGE(ACLNN_ERR_INNER, "The dim [%ld] in shape is less than 0", dim);
                        return ACLNN_ERR_INNER;
                    }
                    aicpuops::TensorShape_Dim* aicpuDims = shape_tensor->add_dim();
                    AICPU_ASSERT_NOTNULL_RETVAL(aicpuDims);
                    aicpuDims->set_size(dim);
                }
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_DATA_TYPE: {
            std::vector<ge::DataType> geTypeList;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::vector<ge::DataType>>(geTypeList));
            auto attrList = attrValue.mutable_array();
            AICPU_ASSERT_NOTNULL_RETVAL(attrList);
            for (ge::DataType& geType : geTypeList) {
                attrList->add_type(static_cast<ge::DataType>(geType));
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_INT: {
            std::vector<int64_t> listIntValue;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::vector<int64_t>>(listIntValue));
            auto attrList = attrValue.mutable_array();
            AICPU_ASSERT_NOTNULL_RETVAL(attrList);
            for (auto value : listIntValue) {
                attrList->add_i(value);
            }
            break;
        }
        case GeAttrValue::ValueType::VT_LIST_STRING: {
            std::vector<std::string> listStringValue;
            AICPU_ASSERT_GE_SUCCESS(geAttrValue.GetValue<std::vector<std::string>>(listStringValue));
            auto attrList = attrValue.mutable_array();
            AICPU_ASSERT_NOTNULL_RETVAL(attrList);
            for (std::string value : listStringValue) {
                attrList->add_s(value);
            }
            break;
        }
        default: {
            OP_LOGW("op [%s] Currently can not support attr value type of [%d].", opType.c_str(), geValueType);
            return OK;
        }
    }
    return OK;
}
using CCAttrValueMap = google::protobuf::Map<string, aicpuops::AttrValue>;
aclnnStatus BuildAicpuNodeDef(const std::string& opType, const FVector<const aclTensor*>& inputs,
                              const FVector<aclTensor*>& outputs, const AicpuAttrs& attrs, aicpuops::NodeDef& nodeDef)
{
    nodeDef.set_op(opType);
    for (auto input : inputs) {
        aicpuops::Tensor* inputTensor = nodeDef.add_inputs();
        AICPU_ASSERT_NOTNULL_RETVAL(inputTensor);
        AICPU_ASSERT_OK_RETVAL(UpdateProtoTensor(input, inputTensor));
    }
    for (auto output : outputs) {
        aicpuops::Tensor* outputTensor = nodeDef.add_outputs();
        AICPU_ASSERT_NOTNULL_RETVAL(outputTensor);
        AICPU_ASSERT_OK_RETVAL(UpdateProtoTensor(output, outputTensor));
    }
    for (auto iter = attrs.cbegin(); iter != attrs.cend(); ++iter) {
        const std::string& attrName = iter->first;
        aicpuops::AttrValue attrValue;
        const GeAttrValue::ValueType geValueType = (iter->second).GetValueType();
        OP_LOGD("Get attr:[%s] value from op [%s], geValueType is [%d].", attrName.c_str(), opType.c_str(),
                geValueType);
        if (GetAttrValueFromGeAttrValue(opType, iter->second, attrValue) != OK) {
            OP_LOGE(ACLNN_ERR_INNER, "GetAttrValueFromGe attrName[%s] for op[%s] failed.", attrName.c_str(),
                    opType.c_str());
            return ACLNN_ERR_INNER;
        }
        auto proto_attrs = nodeDef.mutable_attrs();
        AICPU_ASSERT_NOTNULL_RETVAL(proto_attrs);
        auto pair = proto_attrs->insert(CCAttrValueMap::value_type(attrName, attrValue));
        if (!pair.second) {
            OP_LOGE(ACLNN_ERR_INNER, "Node [%s] insert attr [%s] to nodeDef failed.", opType.c_str(), attrName.c_str());
            return ACLNN_ERR_INNER;
        }
    }
    return OK;
}

aclnnStatus SerializeNodeDefToBuffer(const aicpuops::NodeDef& nodeDef, const std::string& opType, uint8_t* buffer)
{
    google::protobuf::io::ArrayOutputStream arrayStream(buffer, static_cast<int32_t>(nodeDef.ByteSizeLong()));
    google::protobuf::io::CodedOutputStream outputStream(&arrayStream);
    // 保证序列化的接口的一致性
    outputStream.SetSerializationDeterministic(true);
    if (!(nodeDef.SerializeToCodedStream(&outputStream))) {
        OP_LOGE(ACLNN_ERR_INNER, "The serialization from nodedef probuf to str failed, op[%s].", opType.c_str());
        return ACLNN_ERR_INNER;
    }
    return OK;
}
} // namespace

void AicpuArgsHandler::GetDeviceCacheAddr(void*& deviceAddr, aclOpExecutor* executor, const uint64_t deviceCacheOffset)
{
    if (deviceCache_ == nullptr) {
        deviceCache_ = op::internal::ValueToPtr(op::internal::PtrToValue(executor->GetWorkspaceAddr()) +
                                                deviceCacheOffset);
    }
    deviceAddr = op::internal::ValueToPtr(op::internal::PtrToValue(deviceCache_) + deviceCacheSize_);
    OP_LOGD("alloc deviceCache_ success. deviceCacheSize_:%lu", static_cast<uint64_t>(deviceCacheSize_));
}

aclnnStatus AicpuArgsHandler::UpdateIoAddr(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                                           const aclrtStream stream, aclOpExecutor* executor,
                                           const uint64_t deviceExtMemSize, const uint64_t deviceCacheOffset)
{
    uint64_t* ioAddrs = reinterpret_cast<uint64_t*>(hostBuffer_.get() + ioAddrOffset_);
    for (size_t i = 0U; i < inputs.size(); i++) {
        if (inputs[i] == nullptr) {
            *ioAddrs = 0;
            ioAddrs++;
            continue;
        }
        if ((inputs[i]->GetPlacement() == op::TensorPlacement::kOnHost)) {
            auto hostTensorSize = GetSizeInBytes(inputs[i]->GetOriginalShape().GetShapeSize(),
                                                 inputs[i]->GetDataType());
            AICPU_ASSERT_TRUE_RETVAL(hostTensorSize >= 0);
            const size_t alignSize = RoundUp(static_cast<uint64_t>(hostTensorSize), GetInputAddrAlignBytes());
            if (hostInputSize_ + alignSize <= kMaxTotalHostLen) {
                AICPU_ASSERT_OK_RETVAL(AddHostInput(i, inputs[i]->GetData(), hostTensorSize, alignSize));
            } else {
                const uint64_t deviceCacheRsvSize = executor->workspaceDeviceAicpuMem_ - deviceExtMemSize;
                AICPU_ASSERT_TRUE_RETVAL(deviceCacheSize_ + alignSize <= deviceCacheRsvSize);
                void* deviceAddr = nullptr;
                GetDeviceCacheAddr(deviceAddr, executor, deviceCacheOffset);
                // AICPU_ASSERT_RTOK_RETVAL(rtMemcpyAsync(deviceAddr, deviceCacheRsvSize - deviceCacheSize_,
                // inputs[i]->GetData(),
                //                                        hostTensorSize, RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
                AICPU_ASSERT_RTOK_RETVAL(aclrtMemcpyAsync(deviceAddr, deviceCacheRsvSize - deviceCacheSize_,
                                                          inputs[i]->GetData(), hostTensorSize,
                                                          ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
                *ioAddrs = op::internal::PtrToValue(deviceAddr);
                deviceCacheSize_ += alignSize;
            }
        } else {
            *ioAddrs = op::internal::PtrToValue(inputs[i]->GetData());
        }
        ioAddrs++;
    }

    for (size_t i = 0U; i < outputs.size(); i++) {
        if ((outputs[i]->GetPlacement() != op::TensorPlacement::kOnDeviceHbm)) {
            OP_LOGE(ACLNN_ERR_INNER, "output[%zu] placement is [%d] not device placement.", i,
                    outputs[i]->GetPlacement());
            return ACLNN_ERR_INNER;
        }
        *ioAddrs = op::internal::PtrToValue(outputs[i]->GetData());
        ioAddrs++;
    }
    return OK;
}

aclnnStatus AicpuArgsHandler::MallocMem()
{
    hostBuffer_ = std::make_unique<uint8_t[]>(bufferSize_);
    AICPU_ASSERT_NOTNULL_RETVAL(hostBuffer_);
    args_.args = hostBuffer_.get();
    args_.argsSize = bufferSize_;
    args_.hostInputInfoNum = 0U;
    args_.soNameAddrOffset = soNameOffset_;
    args_.kernelNameAddrOffset = kernelNameOffset_;
    return OK;
}

void AicpuArgsHandler::ResetHostInputInfo()
{
    hostInputSize_ = 0U;
    deviceCacheSize_ = 0U;
    deviceCache_ = nullptr;
    hostInputInfo_.clear();
    args_.hostInputInfoNum = 0U;
    args_.hostInputInfoPtr = hostInputInfo_.data();
}

aclnnStatus AicpuArgsHandler::AddHostInput(const size_t idx, void* data, const size_t srcSize, const size_t alignSize)
{
    AICPU_ASSERT_TRUE_RETVAL(hostInputSize_ + alignSize <= kMaxTotalHostLen);
    const size_t remainSize = kMaxTotalHostLen - hostInputSize_;
    auto hostDataOffset = hostMemOffset_ + hostInputSize_;
    auto hostDataAddr = hostBuffer_.get() + hostDataOffset;
    if (srcSize > 0U) {
        AICPU_ASSERT_TRUE_RETVAL(memcpy_s(hostDataAddr, remainSize, data, srcSize) == EOK);
    }

    args_.hostInputInfoNum += 1U;

    auto hostAddrOffset = ioAddrOffset_ + idx * sizeof(void*);
    hostInputInfo_.emplace_back(
        aclrtPlaceHolderInfo({static_cast<uint32_t>(hostAddrOffset), static_cast<uint32_t>(hostDataOffset)}));
    args_.hostInputInfoPtr = hostInputInfo_.data();
    AICPU_ASSERT_TRUE_RETVAL(args_.hostInputInfoNum == hostInputInfo_.size());

    hostInputSize_ += alignSize;
    return OK;
}

// for CC task
// extInfoAddr may host or device
aclnnStatus AicpuCCArgsHandler::SetHostArgs(const std::string& argData, const size_t extInfoSize)
{
    AICPU_ASSERT_OK_RETVAL(memcpy_s(hostBuffer_.get(), bufferSize_, argData.data(), argData.size()));
    auto aicpuParamHead = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(hostBuffer_.get());

    AICPU_ASSERT_TRUE_RETVAL(aicpuParamHead->ioAddrNum == ioNum_);
    const auto miniLen = sizeof(aicpu::AicpuParamHead) + ioNum_ * sizeof(void*);
    AICPU_ASSERT_TRUE_RETVAL(aicpuParamHead->length >= miniLen);
    ioAddrOffset_ = sizeof(aicpu::AicpuParamHead);

    aicpuParamHead->extInfoLength = extInfoSize;

    return OK;
}

aclnnStatus AicpuCCArgsHandler::SetOffsetArgs()
{
    ResetHostInputInfo();
    aicpu::AicpuParamHead paramHead;
    if (!needDeviceExt_) {
        auto extInfoAddrOffset = PtrToValue(&(paramHead.extInfoAddr)) - PtrToValue(&paramHead);
        args_.hostInputInfoNum = 1U;
        hostInputInfo_.emplace_back(
            aclrtPlaceHolderInfo({static_cast<uint32_t>(extInfoAddrOffset), static_cast<uint32_t>(extInfoOffset_)}));
    }
    args_.hostInputInfoPtr = hostInputInfo_.data();
    return OK;
}

aclnnStatus AicpuCCArgsHandler::GenCCArgs(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                                          const AicpuAttrs& attrs, std::string& taskInfo) const
{
    uint32_t paramLenth = static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead));
    size_t inputSize = inputs.size();
    size_t outputSize = outputs.size();
    uint32_t ioAddrsNum = static_cast<uint32_t>(inputSize + outputSize);
    uint32_t ioAddrsSize = ioAddrsNum * static_cast<uint32_t>(sizeof(uint64_t));
    paramLenth += ioAddrsSize;
    paramLenth += static_cast<uint32_t>(sizeof(uint32_t));
    // 序列化nodedef
    aicpuops::NodeDef nodeDef;
    AICPU_ASSERT_OK_RETVAL(BuildAicpuNodeDef(opType_, inputs, outputs, attrs, nodeDef));
    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(nodeDef.ByteSizeLong());
    AICPU_ASSERT_OK_RETVAL(SerializeNodeDefToBuffer(nodeDef, opType_, buffer.get()));
    uint32_t nodeDefLen = static_cast<uint32_t>(nodeDef.ByteSizeLong());
    if (AddOverflow(paramLenth, nodeDefLen, paramLenth)) {
        OP_LOGE(ACLNN_ERR_INNER, "Added overflow when paramLenth is [%u], nodeDefLen is [%u]", paramLenth, nodeDefLen);
        return ACLNN_ERR_INNER;
    }
    // Create task_args: AicpuParamHead + ioAddrs + customizedAttr
    // extend info length 和 addr 会在后面刷新一下
    uint64_t extInfoAddrs = 0UL;
    aicpu::AicpuParamHead paramHead = {paramLenth, ioAddrsNum, 0, extInfoAddrs};
    taskInfo.append(reinterpret_cast<const char*>(&paramHead), sizeof(aicpu::AicpuParamHead));
    for (size_t i = 0; i < inputSize; i++) {
        const uint64_t offset = 0UL;
        taskInfo.append(reinterpret_cast<const char*>(&offset), sizeof(uint64_t));
    }
    for (size_t i = 0; i < outputSize; i++) {
        const uint64_t offset = 0UL;
        taskInfo.append(reinterpret_cast<const char*>(&offset), sizeof(uint64_t));
    }

    taskInfo.append(reinterpret_cast<const char*>(&nodeDefLen), sizeof(uint32_t));
    taskInfo.append(reinterpret_cast<const char*>(buffer.get()), nodeDefLen);
    return OK;
}

aclnnStatus AicpuCCArgsHandler::BuildCCArgs(const std::string& argData, const std::string& kernelName,
                                            const std::string& soName, const size_t extInfoSize)
{
    const size_t argSize = argData.size();
    const size_t kernelNameSize = kernelName.size() + 1;
    const size_t soNameSize = soName.size() + 1;

    extInfoSize_ = extInfoSize;
    kernelNameOffset_ = argSize + ioNum_ * sizeof(void*);
    soNameOffset_ = kernelNameOffset_ + kernelNameSize;
    extInfoOffset_ = soNameOffset_ + soNameSize;
    hostMemOffset_ = extInfoOffset_ + extInfoSize;
    bufferSize_ = hostMemOffset_ + kMaxTotalHostLen;

    AICPU_ASSERT_OK_RETVAL(MallocMem());
    AICPU_ASSERT_OK_RETVAL(SetHostArgs(argData, extInfoSize));

    AICPU_ASSERT_OK_RETVAL(
        memcpy_s(hostBuffer_.get() + kernelNameOffset_, kernelNameSize, kernelName.data(), kernelNameSize));
    AICPU_ASSERT_OK_RETVAL(memcpy_s(hostBuffer_.get() + soNameOffset_, soNameSize, soName.data(), soNameSize));
    if (soName == "custom_sub_repository.so") {
        args_.soNameAddrOffset = 0U;
        args_.kernelNameAddrOffset = 0U;
        OP_LOGI("Do not set the soname for custom operator identification.");
    }
    return OK;
}

void AicpuCCArgsHandler::UpdateDeviceExtInfoAddr(void* deviceExtInfoAddr)
{
    AICPU_ASSERT_OK(SetOffsetArgs());
    // deviceExtInfoAddr only valid in dynamic 3th op
    auto aicpuParamHead = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(hostBuffer_.get());
    aicpuParamHead->extInfoAddr = PtrToValue(deviceExtInfoAddr);
}

aclnnStatus AicpuTfArgsHandler::GenTfArgs(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                                          const AicpuAttrs& attrs, STR_FWK_OP_KERNEL& fwkOpKernel,
                                          std::string& taskInfo) const
{
    // Step1 : define the task api struct.
    fwkOpKernel.fwkKernelType = FMK_KERNEL_TYPE_TF;
    aicpu::FWKAdapter::FWKOperateParam* strTfKernel = &(fwkOpKernel.fwkKernelBase.fwk_kernel);
    strTfKernel->opType = aicpu::FWKAdapter::FWK_ADPT_KERNEL_RUN;

    // Step2 : Build the StrFWKKernel
    aicpu::FWKAdapter::KernelRunParam kernelRunParam{};
    AICPU_ASSERT_OK_RETVAL(BuildKernelRunParam(inputs, outputs, (AicpuTaskSpace*)space_, kernelRunParam));
    strTfKernel->inputOutputLen = static_cast<int64_t>(kernelRunParam.ByteSizeLong());
    std::unique_ptr<uint8_t[]> kernelRunParamBuffer = std::make_unique<uint8_t[]>(kernelRunParam.ByteSizeLong());
    AICPU_ASSERT_OK_RETVAL(SerializeKernelRunParamToBuffer(kernelRunParam, kernelRunParamBuffer.get()));
    taskInfo.append(reinterpret_cast<const char*>(kernelRunParamBuffer.get()), kernelRunParam.ByteSizeLong());

    // Step3 : build the tf's nodeDef
    domi::tensorflow::NodeDef nodeDef;
    nodeDef.set_name(opType_);

    for (size_t i = 0U; i < inputs.size(); ++i) {
        std::string inputName = "input_" + std::to_string(i);
        nodeDef.add_input(inputName);
    }
    AICPU_ASSERT_OK_RETVAL(ParseAttr(inputs, attrs, opType_, nodeDef));
    std::unique_ptr<uint8_t[]> nodeDefBytes = std::make_unique<uint8_t[]>(nodeDef.ByteSizeLong());
    AICPU_ASSERT_OK_RETVAL(SerializeNodedefToBuffer(nodeDef, nodeDefBytes.get(), nodeDef.ByteSizeLong()));

    AICPU_ASSERT_NOTNULL_RETVAL(nodeDefBytes.get());
    strTfKernel->funDefLibLen = 0; // op api no need func def
    strTfKernel->nodeDefLen = nodeDef.ByteSizeLong();
    strTfKernel->inputOutputBuf = 0;
    strTfKernel->nodeDefBuf = strTfKernel->inputOutputBuf + strTfKernel->inputOutputLen;
    taskInfo.append(reinterpret_cast<const char*>(nodeDefBytes.get()), nodeDef.ByteSizeLong());
    return OK;
}

aclnnStatus AicpuTfArgsHandler::SetOffsetArgs()
{
    ResetHostInputInfo();
    STR_FWK_OP_KERNEL fwkOpKernel = {};
    auto workspace_baseAddrOffset = PtrToValue(&(fwkOpKernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr)) -
                                    PtrToValue(&fwkOpKernel);
    auto ioAddrOffset = PtrToValue(&(fwkOpKernel.fwkKernelBase.fwk_kernel.inputOutputAddr)) - PtrToValue(&fwkOpKernel);
    args_.hostInputInfoNum = 2U; // workspace & ioAddr
    hostInputInfo_.emplace_back(aclrtPlaceHolderInfo(
        {static_cast<uint32_t>(workspace_baseAddrOffset), static_cast<uint32_t>(taskInfoOffset_)}));
    hostInputInfo_.emplace_back(
        aclrtPlaceHolderInfo({static_cast<uint32_t>(ioAddrOffset), static_cast<uint32_t>(ioAddrOffset_)}));

    if (!needDeviceExt_) {
        auto extInfoAddrOffset = PtrToValue(&(fwkOpKernel.fwkKernelBase.fwk_kernel.extInfoAddr)) -
                                 PtrToValue(&fwkOpKernel);
        args_.hostInputInfoNum += 1U;
        hostInputInfo_.emplace_back(
            aclrtPlaceHolderInfo({static_cast<uint32_t>(extInfoAddrOffset), static_cast<uint32_t>(extInfoOffset_)}));
    }
    args_.hostInputInfoPtr = hostInputInfo_.data();
    return OK;
}

aclnnStatus AicpuTfArgsHandler::BuildTfArgs(STR_FWK_OP_KERNEL& fwkOpKernel, const std::string& taskInfo,
                                            const size_t extInfoSize)
{
    const size_t argSize = sizeof(STR_FWK_OP_KERNEL);
    const size_t taskInfoSize = taskInfo.size();
    const size_t ioAddrSize = ioNum_ * sizeof(void*);

    extInfoSize_ = extInfoSize;
    taskInfoOffset_ = argSize;
    ioAddrOffset_ = taskInfoOffset_ + taskInfoSize;
    extInfoOffset_ = ioAddrOffset_ + ioAddrSize;
    hostMemOffset_ = extInfoOffset_ + extInfoSize;
    soNameOffset_ = hostMemOffset_ + kMaxTotalHostLen;
    kernelNameOffset_ = soNameOffset_ + sizeof(void*);
    bufferSize_ = kernelNameOffset_ + sizeof(void*);

    // tf input addr must align 64 bytes
    alignBytes_ = 64U;

    AICPU_ASSERT_OK_RETVAL(MallocMem());

    // update extInfo, kernel_id
    fwkOpKernel.fwkKernelBase.fwk_kernel.extInfoLen = extInfoSize;
    fwkOpKernel.fwkKernelBase.fwk_kernel.kernelID = AicpuExtInfoHandler::GenerateKernelId();
    fwkOpKernel.fwkKernelBase.fwk_kernel.opType = aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_RUN_NO_SESS;
    fwkOpKernel.fwkKernelBase.fwk_kernel.sessionID = std::numeric_limits<uint64_t>::max();
    fwkOpKernel.fwkKernelBase.fwk_kernel.stepIDAddr = 0;

    AICPU_ASSERT_OK_RETVAL(
        memcpy_s(hostBuffer_.get(), sizeof(STR_FWK_OP_KERNEL), &fwkOpKernel, sizeof(STR_FWK_OP_KERNEL)));
    AICPU_ASSERT_OK_RETVAL(memcpy_s(hostBuffer_.get() + taskInfoOffset_, taskInfoSize, taskInfo.data(), taskInfoSize));
    return OK;
}

void AicpuTfArgsHandler::UpdateDeviceExtInfoAddr(void* deviceExtInfoAddr)
{
    AICPU_ASSERT_OK(SetOffsetArgs());
    // deviceExtInfoAddr only valid in dynamic 3th op
    auto fwkOpKernel = PtrToPtr<uint8_t, STR_FWK_OP_KERNEL>(hostBuffer_.get());
    fwkOpKernel->fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(deviceExtInfoAddr);
}
} // namespace internal
} // namespace op
