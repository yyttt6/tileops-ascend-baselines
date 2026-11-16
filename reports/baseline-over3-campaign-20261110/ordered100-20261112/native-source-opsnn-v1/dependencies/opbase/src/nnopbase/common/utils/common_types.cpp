/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <securec.h>
#include <graph/utils/type_utils.h>
#include <stdexcept>
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "bridge_graph.h"
#include "bridge_pool.h"
#include "type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_arg_def.h"
#include "opdev/common_types.h"
#include "nnopbase_error_msg.h"
using namespace std;

aclStorage::aclStorage(void* addr) : addr_(addr) {}

aclStorage::aclStorage(bool fromWorkspace) : fromWorkspace_(fromWorkspace) {}

aclStorage::aclStorage(void* addr, bool fromWorkspace) : addr_(addr), fromWorkspace_(fromWorkspace) {}

aclStorage::~aclStorage() { op::internal::FreeExtendTensorImpl(extend_); }

void* aclStorage::GetAddr() const { return addr_; }

void aclStorage::SetAddr(void* addr) { addr_ = addr; }

void aclStorage::SetWorkspaceOffset(uint64_t offset) { workspaceOffset_ = offset; }

uint64_t aclStorage::GetWorkspaceOffset() const { return workspaceOffset_; }

void aclStorage::SetFromWorkspace(bool fromWorkspace) { fromWorkspace_ = fromWorkspace; }

bool aclStorage::IsFromWorkspace() const { return fromWorkspace_; }

void aclStorage::SetExtend(void* extend) { extend_ = extend; }

void* aclStorage::GetExtend() const { return extend_; }

void aclStorage::SetStorageOffset(int64_t offset) { storageOffset_ = offset; }

int64_t aclStorage::GetStorageOffset() const { return storageOffset_; }

template <typename T>
aclArray<T>::~aclArray()
{
    if (value_ != nullptr) {
        op::internal::DeAllocate(value_);
    }
}

template <typename T>
T& aclArray<T>::operator[](uint64_t i)
{
    return value_[i];
}
template <typename T>
const T& aclArray<T>::operator[](uint64_t i) const
{
    return value_[i];
}
template <typename T>
uint64_t aclArray<T>::Size() const
{
    return size_;
}
template <typename T>
const T* aclArray<T>::GetData() const
{
    return value_;
}
template <typename T>
ge::AscendString aclArray<T>::ToString() const
{
    std::string res = "[";
    for (uint64_t i = 0; i < size_; i++) {
        if constexpr (std::is_same<op::fp16_t, typename std::decay<T>::type>::value) {
            res += std::to_string(value_[i].toFloat());
        } else if constexpr (std::is_same<op::bfloat16, typename std::decay<T>::type>::value) {
            res += std::to_string(float(value_[i]));
        } else {
            res += std::to_string(value_[i]);
        }
        if (i != size_ - 1) {
            res += ", ";
        }
    }
    res += "]";
    return ge::AscendString(res.c_str());
}
template <typename T>
aclArray<T>::aclArray(const T* value, uint64_t size)
{
    if (value != nullptr && size != 0) {
        void* base = op::internal::Allocate(size * sizeof(T));
        OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclArray allocate failed."), throw std::bad_alloc());
        size_ = size;
        value_ = new (base) T[size];
        if constexpr (std::is_trivial_v<T>) {
            OP_CHECK(memcpy_s(value_, size * sizeof(T), value, size * sizeof(T)) == EOK,
                     OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
                     throw std::runtime_error("aclArray<T>::aclArray memcpy runtime error."));
        } else {
            for (uint64_t i = 0; i < size; i++) {
                value_[i] = value[i];
            }
        }
    }
}

aclTensor::aclTensor(op::DataType dataType, op::Format storageFormat, op::Format originFormat)
    : storage_(new (std::nothrow) aclStorage(true)),
      viewOffset_(0),
      viewStrides_(),
      viewShape_(),
      viewFormat_(originFormat)
{
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    op::StorageShape shape;
    op::StorageFormat format;
    format.SetStorageFormat(storageFormat);
    format.SetOriginFormat(originFormat);
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(shape, format, op::TensorPlacement::kOnDeviceHbm, dataType, nullptr);
}

aclTensor::aclTensor(const op::Shape& shape, op::DataType dataType, op::Format format, void* tensorDataAddr)
    : aclTensor(shape, shape, dataType, format, format, tensorDataAddr)
{}

aclTensor::aclTensor(const op::Shape& storageShape, const op::Shape& originShape, op::DataType dataType,
                     op::Format storageFormat, op::Format originFormat, void* tensorDataAddr)
    : storage_(new (std::nothrow) aclStorage(tensorDataAddr, true)),
      viewOffset_(0),
      viewStrides_(),
      viewShape_(originShape),
      viewFormat_(originFormat)
{
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    op::StorageShape gertShape;
    op::StorageFormat gertFormat;
    gertShape.MutableStorageShape() = storageShape;
    gertShape.MutableOriginShape() = originShape;
    gertFormat.SetStorageFormat(storageFormat);
    gertFormat.SetOriginFormat(originFormat);
    op::ToContiguousStrides(viewShape_, viewStrides_);

    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(gertShape, gertFormat, op::TensorPlacement::kOnDeviceHbm, dataType, tensorDataAddr);
}

aclTensor::aclTensor(const op::Shape& shape, op::DataType dataType, op::Format format)
    : aclTensor(shape, shape, dataType, format, format)
{}

aclTensor::aclTensor(const op::Shape& storageShape, const op::Shape& originShape, op::DataType dataType,
                     op::Format storageFormat, op::Format originFormat)
    : viewOffset_(0), viewStrides_(), viewShape_(originShape), viewFormat_(originFormat)
{
    auto typeSize = op::TypeSize(dataType);
    int64_t size = storageShape.GetShapeSize();
    void* data = op::internal::Allocate(size * typeSize);
    OP_CHECK(data != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    void* dataAddr = new (data) int64_t[size];
    storage_ = new (std::nothrow) aclStorage(dataAddr, false);
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    op::ToContiguousStrides(viewShape_, viewStrides_);

    op::StorageShape gertShape;
    op::StorageFormat gertFormat;
    gertShape.MutableStorageShape() = storageShape;
    gertShape.MutableOriginShape() = originShape;
    gertFormat.SetStorageFormat(storageFormat);
    gertFormat.SetOriginFormat(originFormat);
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(gertShape, gertFormat, op::TensorPlacement::kOnHost, dataType, dataAddr);
}

aclTensor::aclTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride,
                     int64_t offset, const aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                     void* tensorDataAddr)
    : viewOffset_(offset), viewStrides_(), viewShape_(), viewFormat_(static_cast<op::Format>(format))
{
    op::ToShape(viewDims, viewDimsNum, viewShape_);
    if (stride != nullptr) {
        viewStrides_.assign(stride, stride + viewDimsNum);
    } else {
        op::ToContiguousStrides(viewShape_, viewStrides_);
    }

    op::StorageShape storageShape;
    op::StorageFormat storageFormat;
    op::ToShape(storageDims, storageDimsNum, storageShape.MutableStorageShape());
    op::ToShape(storageDims, storageDimsNum, storageShape.MutableOriginShape());
    storageFormat.SetStorageFormat(op::ToOpFormat(format));
    storageFormat.SetOriginFormat(op::ToOpFormat(format));

    OP_CHECK(!op::IsPrivateFormat(viewFormat_),
             OP_LOGW("the private format[%s], should not use this constructor.", op::ToString(viewFormat_).GetString()),
             ;);

    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(storageShape, storageFormat, op::TensorPlacement::kOnDeviceHbm,
                                    op::ToOpDataType(dataType), tensorDataAddr);
    storage_ = new (std::nothrow) aclStorage(tensorDataAddr);
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
}

aclTensor::aclTensor(const aclTensor& other, const op::Shape& shape, int64_t offset)
    : storage_(other.storage_),
      viewOffset_(offset),
      viewStrides_(),
      viewShape_(shape),
      viewFormat_(other.viewFormat_),
      isView_(true)
{
    op::StorageShape storageShape;
    storageShape.MutableStorageShape() = shape;
    storageShape.MutableOriginShape() = shape;
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(storageShape, other.tensor_->GetFormat(), other.tensor_->GetPlacement(),
                                    other.GetDataType(), other.tensor_->GetTensorData().GetAddr());
    op::ToContiguousStrides(viewShape_, viewStrides_);
}

aclTensor::aclTensor(const aclTensor& other, const op::Shape& oriShape, const op::Shape& storageShape,
                     const op::Strides& oriStride, int64_t offset)
    : storage_(other.storage_),
      viewOffset_(offset),
      viewStrides_(oriStride),
      viewShape_(oriShape),
      viewFormat_(other.viewFormat_),
      isView_(true)
{
    op::StorageShape storageShapeLocal;
    storageShapeLocal.MutableStorageShape() = storageShape;
    storageShapeLocal.MutableOriginShape() = oriShape;
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(storageShapeLocal, other.tensor_->GetFormat(), other.tensor_->GetPlacement(),
                                    other.GetDataType(), other.tensor_->GetTensorData().GetAddr());
    auto& geStride = tensor_->MutableStride();
    geStride.SetDimNum(oriStride.size());
    for (size_t i = 0; i < oriStride.size(); i++) {
        geStride.SetStride(i, oriStride[i]);
    }
}

aclTensor::aclTensor(const aclIntArray* value, op::DataType dataType)
    : aclTensor(value->GetData(), value->Size(), dataType)
{}

aclTensor::aclTensor(const aclBoolArray* value, op::DataType dataType)
    : aclTensor(value->GetData(), value->Size(), dataType)
{}

aclTensor::aclTensor(const aclFloatArray* value, op::DataType dataType)
    : aclTensor(value->GetData(), value->Size(), dataType)
{}

aclTensor::aclTensor(const aclFp16Array* value, op::DataType dataType)
    : aclTensor(value->GetData(), value->Size(), dataType)
{}

aclTensor::aclTensor(const aclBf16Array* value, op::DataType dataType)
    : aclTensor(value->GetData(), value->Size(), dataType)
{}

aclTensor::aclTensor(const aclScalar* value, op::DataType dataType)
    : viewOffset_(0), viewStrides_(), viewShape_({}), viewFormat_(op::Format::FORMAT_ND), isView_(false)
{
    auto typeSize = op::TypeSize(dataType);
    void* dataAddr = op::internal::Allocate(typeSize);
    OP_CHECK(dataAddr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    storage_ = new (std::nothrow) aclStorage(dataAddr, false);
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    op::StorageShape storageShape({}, {});
    op::StorageFormat storageFormat;
    storageFormat.SetStorageFormat(op::Format::FORMAT_ND);
    storageFormat.SetOriginFormat(op::Format::FORMAT_ND);
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(storageShape, storageFormat, op::TensorPlacement::kOnHost, dataType, dataAddr);
    switch (dataType) {
        case op::DataType::DT_FLOAT:
            *static_cast<float*>(dataAddr) = value->ToFloat();
            break;
        case op::DataType::DT_FLOAT16:
            *static_cast<uint16_t*>(dataAddr) = value->ToFp16().val;
            break;
        case op::DataType::DT_BF16:
            *static_cast<uint16_t*>(dataAddr) = value->ToBf16().value;
            break;
        case op::DataType::DT_INT8:
            *static_cast<int8_t*>(dataAddr) = value->ToInt8();
            break;
        case op::DataType::DT_INT16:
            *static_cast<int16_t*>(dataAddr) = value->ToInt16();
            break;
        case op::DataType::DT_UINT16:
            *static_cast<uint16_t*>(dataAddr) = value->ToUint16();
            break;
        case op::DataType::DT_UINT8:
            *static_cast<uint8_t*>(dataAddr) = value->ToUint8();
            break;
        case op::DataType::DT_INT32:
            *static_cast<int32_t*>(dataAddr) = value->ToInt32();
            break;
        case op::DataType::DT_INT64:
            *static_cast<int64_t*>(dataAddr) = value->ToInt64();
            break;
        case op::DataType::DT_UINT32:
            *static_cast<uint32_t*>(dataAddr) = value->ToUint32();
            break;
        case op::DataType::DT_UINT64:
            *static_cast<uint64_t*>(dataAddr) = value->ToUint64();
            break;
        case op::DataType::DT_BOOL:
            *static_cast<bool*>(dataAddr) = value->ToBool();
            break;
        case op::DataType::DT_DOUBLE:
            *static_cast<double*>(dataAddr) = value->ToDouble();
            break;
        case op::DataType::DT_COMPLEX64:
            *static_cast<std::complex<float>*>(dataAddr) = value->ToComplex64();
            break;
        case op::DataType::DT_COMPLEX128:
            *static_cast<std::complex<double>*>(dataAddr) = value->ToComplex128();
            break;
        case op::DataType::DT_FLOAT8_E5M2:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat8E5M2().value;
            break;
        case op::DataType::DT_FLOAT8_E4M3FN:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat8E4M3FN().value;
            break;
        case op::DataType::DT_FLOAT8_E8M0:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat8E8M0().value;
            break;
        case op::DataType::DT_FLOAT6_E3M2:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat6E3M2().value;
            break;
        case op::DataType::DT_FLOAT6_E2M3:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat6E2M3().value;
            break;
        case op::DataType::DT_FLOAT4_E2M1:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat4E2M1().value;
            break;
        case op::DataType::DT_FLOAT4_E1M2:
            *static_cast<uint8_t*>(dataAddr) = value->ToFloat4E1M2().value;
            break;
        case op::DataType::DT_HIFLOAT8:
            *static_cast<uint8_t*>(dataAddr) = value->ToHiFloat8().value;
            break;
        default:
            OP_CHECK(memset_s(dataAddr, typeSize, 0, typeSize) == EOK,
                     OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "memset_s failed."), return);
            std::string dtypeRange = "[DT_FLOAT(0), DT_FLOAT16(1), DT_INT8(2), DT_INT32(3), DT_UINT8(4), "
                                     "DT_INT16(6), DT_UINT16(7), DT_UINT32(8), DT_INT64(9), DT_UINT64(10), "
                                     "DT_DOUBLE(11), DT_BOOL(12), DT_COMPLEX64(16), DT_COMPLEX128(17), DT_BF16(27), "
                                     "DT_HIFLOAT8(34), DT_FLOAT8_E5M2(35), DT_FLOAT8_E4M3FN(36), DT_FLOAT8_E8M0(37), "
                                     "DT_FLOAT6_E3M2(38), DT_FLOAT6_E2M3(39), DT_FLOAT4_E2M1(40), DT_FLOAT4_E1M2(41)]";
            std::string dataTypeStr = std::string(op::ToString(dataType).GetString()) + "(" +
                                      std::to_string(static_cast<int32_t>(dataType)) + ")";
            OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(dataTypeStr.c_str(), dtypeRange.c_str());
            throw std::invalid_argument("dataType not supported.");
            break;
    }
}

aclTensor::~aclTensor()
{
    if (!isView_) {
        if (GetPlacement() == op::TensorPlacement::kOnHost) {
            auto addr = static_cast<char*>(storage_->GetAddr());
            op::internal::DeAllocate(addr);
        }
        delete storage_;
    }
    op::internal::DeAllocate(tensor_);
}

const op::Shape& aclTensor::GetStorageShape() const { return tensor_->GetShape().GetStorageShape(); }

const op::Shape& aclTensor::GetOriginalShape() const { return tensor_->GetShape().GetOriginShape(); }

const op::Shape& aclTensor::GetViewShape() const { return viewShape_; }

op::Format aclTensor::GetStorageFormat() const { return tensor_->GetFormat().GetStorageFormat(); }

op::Format aclTensor::GetOriginalFormat() const { return tensor_->GetFormat().GetOriginFormat(); }

op::Format aclTensor::GetViewFormat() const { return viewFormat_; }

const op::Strides& aclTensor::GetViewStrides() const { return viewStrides_; }

op::Tensor* aclTensor::GetTensor() const { return tensor_; }

void* aclTensor::GetData() const
{
    if (IsFromWorkspace()) {
        return static_cast<char*>(storage_->GetAddr()) + storage_->GetWorkspaceOffset() +
               op::CalcShapeBytes(viewOffset_, tensor_->GetDataType(), true);
    }

    return static_cast<char*>(storage_->GetAddr()) +
           op::CalcShapeBytes(viewOffset_ + GetStorageOffset(), tensor_->GetDataType(), true);
}

int64_t aclTensor::Size() const { return GetStorageShape().GetShapeSize(); }

int64_t aclTensor::Numel() const { return GetViewShape().GetShapeSize(); }

op::DataType aclTensor::GetDataType() const
{
    if (tensor_) {
        return tensor_->GetDataType();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "the aclTensor(%p)'s tensor_ address is %p. please check the memory usage in your code.", this,
                tensor_);
        return op::DataType::DT_MAX;
    }
}

int64_t aclTensor::GetViewOffset() const { return viewOffset_; }

void aclTensor::SetViewOffset(int64_t offset) const { viewOffset_ = offset; }

int64_t aclTensor::GetStorageOffset() const { return storage_->GetStorageOffset(); }

void aclTensor::SetStorageOffset(int64_t offset) { storage_->SetStorageOffset(offset); }

bool aclTensor::IsFromWorkspace() const { return storage_->IsFromWorkspace(); }

void aclTensor::SetStorageAddr(void* addr) const
{
    OP_LOGI("aclTensor: %p, aclStorage: %p, set addr: %p", this, storage_, addr);
    storage_->SetAddr(addr);
    tensor_->MutableTensorData().SetAddr(addr, nullptr);
}

void aclTensor::SetFromWorkspace(bool from_workspace) const { storage_->SetFromWorkspace(from_workspace); }

void aclTensor::SetStorageShape(const op::Shape& shape) const
{
    tensor_->MutableStorageShape() = shape;
    // tensor size need to resize when storage shape update
    tensor_->SetSize(op::CalcShapeBytes(tensor_->GetShapeSize(), tensor_->GetDataType()));
}

void aclTensor::SetOriginalShape(const op::Shape& shape) const { tensor_->MutableOriginShape() = shape; }

void aclTensor::SetViewShape(const op::Shape& shape)
{
    viewShape_ = shape;
    op::ToContiguousStrides(viewShape_, viewStrides_);
}

void aclTensor::SetStorageFormat(op::Format format) { tensor_->SetStorageFormat(format); }

void aclTensor::SetOriginalFormat(op::Format format) { tensor_->SetOriginFormat(format); }

void aclTensor::SetViewFormat(op::Format format) { viewFormat_ = format; }

void aclTensor::SetViewStrides(const op::Strides& strides) { viewStrides_ = strides; }

void aclTensor::SetViewStrides(op::Strides&& strides) { viewStrides_ = std::move(strides); }

bool aclTensor::IsView() const { return isView_; }

void aclTensor::SetView(bool is_view) { isView_ = is_view; }

void aclTensor::SetWorkspaceOffset(uint64_t offset) const { storage_->SetWorkspaceOffset(offset); }

void aclTensor::SetExtend(void* extend) const { storage_->SetExtend(extend); }

void* aclTensor::GetExtend() const { return storage_->GetExtend(); }

uint64_t aclTensor::GetWorkspaceOffset() const { return storage_->GetWorkspaceOffset(); }
op::TensorPlacement aclTensor::GetPlacement() const { return tensor_->GetPlacement(); }
void* aclTensor::GetStorageAddr() const { return storage_->GetAddr(); }

const aclStorage* aclTensor::GetStorage() const { return storage_; }

bool aclTensor::IsEmpty() const
{
    bool isEmpty = false;
    for (size_t i = 0; i < viewShape_.GetDimNum(); i++) {
        if (viewShape_[i] == 0) {
            isEmpty = true;
            break;
        }
    }
    return isEmpty;
}

void aclTensor::InitTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride,
                           int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorDataAddr)
{
    if (viewDims && viewDimsNum) {
        op::ToShape(viewDims, viewDimsNum, viewShape_);
    }
    viewFormat_ = static_cast<op::Format>(format);
    if (stride != nullptr) {
        viewStrides_.assign(stride, stride + viewDimsNum);
    } else {
        op::ToContiguousStrides(viewShape_, viewStrides_);
    }

    viewOffset_ = offset;

    if (op::IsPrivateFormat(viewFormat_)) {
        OP_LOGW("the private format[%s], should not use this constructor.", op::ToString(viewFormat_).GetString());
    }

    if (storage_ && tensorDataAddr) {
        storage_->SetAddr(tensorDataAddr);
    }

    if (tensor_) {
        if (storageDims && storageDimsNum) {
            op::Shape shape;
            op::ToShape(storageDims, storageDimsNum, shape);
            tensor_->MutableStorageShape() = shape;
            tensor_->MutableOriginShape() = shape;
        }

        op::Format opFormat = op::ToOpFormat(format);
        tensor_->SetStorageFormat(opFormat);
        tensor_->SetOriginFormat(opFormat);
        tensor_->SetDataType(op::ToOpDataType(dataType));

        if (tensorDataAddr) {
            tensor_->MutableTensorData().SetAddr(tensorDataAddr, nullptr);
        }
    }
}

template <typename T, typename dataType>
static void SetDataByDataType(int64_t index, void* dataAddr, const T value)
{
    dataType* tmpDataAddr = static_cast<dataType*>(dataAddr);
    if constexpr (op::internal::IsCustomFloat<typename std::decay<T>::type>::value) {
        // For custom float types, convert through double to avoid ambiguity
        *(tmpDataAddr + index) = static_cast<dataType>(static_cast<double>(value));
    } else {
        *(tmpDataAddr + index) = static_cast<dataType>(value);
    }
}

template <typename T>
static void SetDataByBool(int64_t index, void* dataAddr, const T value)
{
    auto tmpDataAddr = static_cast<bool*>(dataAddr);
    if constexpr (std::is_floating_point<T>::value || std::is_same<op::fp16_t, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float8E5M2, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float8E4M3FN, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float8E8M0, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float6E3M2, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float6E2M3, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float4E2M1, typename std::decay<T>::type>::value ||
                  std::is_same<op::Float4E1M2, typename std::decay<T>::type>::value ||
                  std::is_same<op::HiFloat4, typename std::decay<T>::type>::value ||
                  std::is_same<op::HiFloat8, typename std::decay<T>::type>::value) {
        *(tmpDataAddr + index) = std::abs(static_cast<float>(value)) >= std::numeric_limits<float>::epsilon();
    } else {
        *(tmpDataAddr + index) = static_cast<bool>(value);
    }
}

template <typename T>
void aclTensor::SetData(int64_t index, const T value, op::DataType dataType)
{
    if (this->GetPlacement() == op::TensorPlacement::kOnHost) {
        void* dataAddr = this->GetStorageAddr();
        switch (dataType) {
            case op::DataType::DT_FLOAT: {
                SetDataByDataType<T, float>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_FLOAT16: {
                SetDataByDataType<T, op::fp16_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_BF16: {
                SetDataByDataType<T, op::bfloat16>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_INT8: {
                SetDataByDataType<T, int8_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_INT16: {
                SetDataByDataType<T, int16_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_UINT16: {
                SetDataByDataType<T, uint16_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_UINT8: {
                SetDataByDataType<T, uint8_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_INT32: {
                SetDataByDataType<T, int32_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_INT64: {
                SetDataByDataType<T, int64_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_UINT32: {
                SetDataByDataType<T, uint32_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_UINT64: {
                SetDataByDataType<T, uint64_t>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_DOUBLE: {
                SetDataByDataType<T, double>(index, dataAddr, value);
                break;
            }
            case op::DataType::DT_BOOL: {
                SetDataByBool(index, dataAddr, value);
                break;
            }
            default: {
                std::string dtypeRange = "[DT_FLOAT(0), DT_FLOAT16(1), DT_INT8(2), DT_INT32(3), DT_UINT8(4), "
                                         "DT_INT16(6), DT_UINT16(7), DT_UINT32(8), DT_INT64(9), DT_UINT64(10), "
                                         "DT_DOUBLE(11), DT_BOOL(12), DT_BF16(27)]";
                std::string dataTypeStr = std::string(op::ToString(dataType).GetString()) + "(" +
                                          std::to_string(static_cast<int32_t>(dataType)) + ")";
                OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(dataTypeStr.c_str(), dtypeRange.c_str());
            }
        }
    }
}

template <typename T>
void aclTensor::SetData(const T* value, uint64_t size, op::DataType dataType)
{
    if (this->GetPlacement() == op::TensorPlacement::kOnHost) {
        for (uint64_t i = 0; i < size; i++) {
            SetData(i, value[i], dataType);
        }
    }
}

void aclTensor::SetDataType(op::DataType dataType)
{
    bool needReSize = (op::TypeSize(tensor_->GetDataType()) != op::TypeSize(dataType)) ? true : false;
    tensor_->SetDataType(dataType);
    if (needReSize) {
        tensor_->SetSize(op::CalcShapeBytes(tensor_->GetShapeSize(), dataType));
    }
}

void aclTensor::SetBoolData(const bool* value, uint64_t size, op::DataType dataType) { SetData(value, size, dataType); }

void aclTensor::SetIntData(const int64_t* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloatData(const float* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFp16Data(const op::fp16_t* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetBf16Data(const op::bfloat16* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat8E5M2Data(const op::Float8E5M2* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat8E4M3FNData(const op::Float8E4M3FN* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat8E8M0Data(const op::Float8E8M0* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat6E3M2Data(const op::Float6E3M2* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat6E2M3Data(const op::Float6E2M3* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat4E2M1Data(const op::Float4E2M1* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetFloat4E1M2Data(const op::Float4E1M2* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetHiFloat4Data(const op::HiFloat4* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

void aclTensor::SetHiFloat8Data(const op::HiFloat8* value, uint64_t size, op::DataType dataType)
{
    SetData(value, size, dataType);
}

ge::AscendString aclTensor::ToString() const
{
    std::ostringstream oss;
    oss << GetData();
    std::string devicePtr = oss.str();
    std::string res = std::string("aclTensor(") + "view_shape = " + op::ToString(viewShape_).GetString() +
                      ", view_strides = " + op::ToString(viewStrides_).GetString() +
                      ", view_offset = " + std::to_string(viewOffset_) +
                      ", view_format = " + op::ToString(viewFormat_).GetString() +
                      ", storage_shape = " + op::ToString(GetStorageShape()).GetString() +
                      ", storage_format = " + op::ToString(GetStorageFormat()).GetString() +
                      ", ori_shape = " + op::ToString(GetOriginalShape()).GetString() +
                      ", ori_format = " + op::ToString(GetOriginalFormat()).GetString() +
                      ", data_type = " + op::ToString(GetDataType()).GetString() + ", device_ptr = " + devicePtr + ")";
    return ge::AscendString(res.c_str());
}

aclTensorList::aclTensorList(const aclTensor* const* tensors, uint64_t size)
{
    if (tensors != nullptr && size != 0) {
        this->size_ = size;
        this->tensors_ = new (std::nothrow) aclTensor*[size];
        OP_CHECK(memcpy_s(this->tensors_, size * sizeof(aclTensor*), tensors, size * sizeof(aclTensor*)) == EOK,
                 OP_LOGW("Failed to memcpy in aclTensorList create."),
                 throw std::runtime_error("aclTensorList::aclTensorList memcpy runtime error."));
    }
}

const aclTensor* const* aclTensorList::GetData() const { return tensors_; }

ge::AscendString aclTensorList::ToString() const
{
    std::string s;
    if (tensors_ == nullptr) {
        return ge::AscendString(s.c_str());
    }
    for (uint64_t i = 0; i < size_; i++) {
        aclTensor* t = tensors_[i];
        string splitStr = (i < (size_ - 1)) ? ", " : "";
        if (t) {
            s += t->ToString().GetString();
        } else {
            s += "nullptr";
        }
        s += splitStr;
    }
    return ge::AscendString(s.c_str());
}

uint64_t aclTensorList::Size() const { return size_; }

aclTensorList::~aclTensorList()
{
    for (uint64_t i = 0; i < size_; i++) {
        tensors_[i] = nullptr;
    }

    if (tensors_ != nullptr) {
        delete[] tensors_;
    }
}

aclTensor*& aclTensorList::operator[](uint64_t i) { return tensors_[i]; }

const aclTensor* aclTensorList::operator[](uint64_t i) const { return tensors_[i]; }

aclScalarList::aclScalarList(const aclScalar* const* scalars, uint64_t size)
{
    if (scalars != nullptr && size != 0) {
        this->size_ = size;
        this->scalars_ = new (std::nothrow) aclScalar*[size];
        OP_CHECK(memcpy_s(this->scalars_, size * sizeof(aclScalar*), scalars, size * sizeof(aclScalar*)) == EOK,
                 OP_LOGW("Failed to memcpy in aclScalarList create."),
                 throw std::runtime_error("aclScalarList::aclScalarList memcpy runtime error."));
    }
}

const aclScalar* const* aclScalarList::GetData() const { return scalars_; }

ge::AscendString aclScalarList::ToString() const
{
    std::string s;
    if (scalars_ == nullptr) {
        return ge::AscendString(s.c_str());
    }
    for (uint64_t i = 0; i < size_; i++) {
        aclScalar* t = scalars_[i];
        string splitStr = (i < (size_ - 1)) ? ", " : "";
        if (t) {
            s += t->ToString().GetString();
        } else {
            s += "nullptr";
        }
        s += splitStr;
    }
    return ge::AscendString(s.c_str());
}

uint64_t aclScalarList::Size() const { return size_; }

aclScalarList::~aclScalarList()
{
    for (uint64_t i = 0; i < size_; i++) {
        scalars_[i] = nullptr;
    }

    if (scalars_ != nullptr) {
        delete[] scalars_;
    }
}

aclScalar*& aclScalarList::operator[](uint64_t i) { return scalars_[i]; }

const aclScalar* aclScalarList::operator[](uint64_t i) const { return scalars_[i]; }

template <typename to>
bool aclScalar::CheckOverflows() const
{
    switch (dataType_) {
        case op::DataType::DT_FLOAT:
            return op::internal::Overflows<to>(v.f);
        case op::DataType::DT_FLOAT16:
            return op::internal::Overflows<to>(op::fp16_t(v.ui16).toFloat());
        case op::DataType::DT_BF16:
            return op::internal::Overflows<to>(static_cast<float>(BFloat16()));
        case op::DataType::DT_INT8:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.i8);
            }
        case op::DataType::DT_INT16:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.i16);
            }
        case op::DataType::DT_UINT16:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.ui16);
            }
        case op::DataType::DT_UINT8:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.ui8);
            }
        case op::DataType::DT_INT32:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.i32);
            }
        case op::DataType::DT_INT64:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.i64);
            }
        case op::DataType::DT_UINT32:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.ui32);
            }
        case op::DataType::DT_UINT64:
            if constexpr (!std::is_same<std::complex<double>, typename std::decay<to>::type>::value &&
                          !std::is_same<std::complex<float>, typename std::decay<to>::type>::value) {
                return op::internal::Overflows<to>(v.ui64);
            }
        case op::DataType::DT_BOOL:
            return op::internal::Overflows<to>(bool(v.i8));
        case op::DataType::DT_DOUBLE:
            return op::internal::Overflows<to>(v.d);
        case op::DataType::DT_COMPLEX64:
            return op::internal::Overflows<to>(v.complex64);
        case op::DataType::DT_COMPLEX128:
            return op::internal::Overflows<to>(v.complex128);
        default:
            OP_LOGW("no supported data type[%s].", op::ToString(dataType_).GetString());
            break;
    }
    return true;
}

template <typename T, typename FloatType>
static T ConvertCustomFloatTo(uint8_t bits)
{
    FloatType val(bits, FloatType::FromBits());
    if constexpr (std::is_same<FloatType, typename std::decay<T>::type>::value) {
        return val;
    } else if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
        return std::abs(static_cast<float>(val)) >= std::numeric_limits<float>::epsilon();
    } else {
        return static_cast<T>(static_cast<float>(val));
    }
}

template <typename T>
T aclScalar::To() const
{
    switch (dataType_) {
        case op::DataType::DT_FLOAT:
            if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
                return std::abs(v.f) >= std::numeric_limits<float>::epsilon();
            } else {
                return static_cast<T>(v.f);
            }
        case op::DataType::DT_FLOAT16:
            if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
                return std::abs(op::fp16_t(v.ui16).toFloat()) >= std::numeric_limits<float>::epsilon();
            } else {
                return static_cast<T>(op::fp16_t(v.ui16).toDouble());
            }
        case op::DataType::DT_BF16:
            if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
                return std::abs(static_cast<float>(BFloat16())) >= std::numeric_limits<float>::epsilon();
            } else {
                return static_cast<T>(BFloat16());
            }
        case op::DataType::DT_FLOAT8_E5M2:
            return ConvertCustomFloatTo<T, op::Float8E5M2>(v.ui8);
        case op::DataType::DT_FLOAT8_E4M3FN:
            return ConvertCustomFloatTo<T, op::Float8E4M3FN>(v.ui8);
        case op::DataType::DT_FLOAT8_E8M0:
            return ConvertCustomFloatTo<T, op::Float8E8M0>(v.ui8);
        case op::DataType::DT_FLOAT6_E3M2:
            return ConvertCustomFloatTo<T, op::Float6E3M2>(v.ui8);
        case op::DataType::DT_FLOAT6_E2M3:
            return ConvertCustomFloatTo<T, op::Float6E2M3>(v.ui8);
        case op::DataType::DT_FLOAT4_E2M1:
            return ConvertCustomFloatTo<T, op::Float4E2M1>(v.ui8);
        case op::DataType::DT_FLOAT4_E1M2:
            return ConvertCustomFloatTo<T, op::Float4E1M2>(v.ui8);
        case op::DataType::DT_HIFLOAT8:
            return ConvertCustomFloatTo<T, op::HiFloat8>(v.ui8);
        case op::DataType::DT_INT8:
            return static_cast<T>(v.i8);
        case op::DataType::DT_INT16:
            return static_cast<T>(v.i16);
        case op::DataType::DT_UINT16:
            // fp16_t数据结构，无法直接通过uint16_t转换得到，需要先转为fp32，再转fp16_t
            if constexpr (std::is_same<op::fp16_t, typename std::decay<T>::type>::value) {
                float fval = static_cast<float>(v.ui16);
                return static_cast<T>(fval);
            } else {
                return static_cast<T>(v.ui16);
            }
        case op::DataType::DT_UINT8:
            return static_cast<T>(v.ui8);
        case op::DataType::DT_INT32:
            return static_cast<T>(v.i32);
        case op::DataType::DT_INT64:
            return static_cast<T>(v.i64);
        case op::DataType::DT_UINT32:
            return static_cast<T>(v.ui32);
        case op::DataType::DT_UINT64:
            return static_cast<T>(v.ui64);
        case op::DataType::DT_BOOL:
            return static_cast<T>(v.i8);
        case op::DataType::DT_DOUBLE:
            if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
                return std::abs(v.d) >= std::numeric_limits<double>::epsilon();
            } else {
                return static_cast<T>(v.d);
            }
        case op::DataType::DT_COMPLEX64:
            if constexpr (std::is_same<std::complex<float>, typename std::decay<T>::type>::value) {
                return v.complex64;
            } else if constexpr (std::is_same<std::complex<double>, typename std::decay<T>::type>::value) {
                return std::complex<double>(v.complex64);
            }
            OP_LOGW("data type[DT_COMPLEX64] does not support conversion to non-complex types.");
            return T{};
        case op::DataType::DT_COMPLEX128:
            if constexpr (std::is_same<std::complex<double>, typename std::decay<T>::type>::value) {
                return v.complex128;
            } else if constexpr (std::is_same<std::complex<float>, typename std::decay<T>::type>::value) {
                return static_cast<std::complex<float>>(v.complex128);
            }
            OP_LOGW("data type[DT_COMPLEX128] does not support conversion to non-complex types.");
            return T{};
        default:
            OP_LOGW("data type[%s] does not support To conversion.", op::ToString(dataType_).GetString());
            return T{};
    }
}

ge::AscendString aclScalar::ToStr() const
{
    switch (dataType_) {
        case op::DataType::DT_FLOAT:
            return ge::AscendString(std::to_string(v.f).c_str());
        case op::DataType::DT_FLOAT16:
            return ge::AscendString(std::to_string(op::fp16_t(v.ui16).toFloat()).c_str());
        case op::DataType::DT_BF16:
            return ge::AscendString(std::to_string(static_cast<float>(BFloat16())).c_str());
        case op::DataType::DT_FLOAT8_E5M2:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float8E5M2(v.ui8, op::Float8E5M2::FromBits()))).c_str());
        case op::DataType::DT_FLOAT8_E4M3FN:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float8E4M3FN(v.ui8, op::Float8E4M3FN::FromBits()))).c_str());
        case op::DataType::DT_FLOAT8_E8M0:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float8E8M0(v.ui8, op::Float8E8M0::FromBits()))).c_str());
        case op::DataType::DT_FLOAT6_E3M2:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float6E3M2(v.ui8, op::Float6E3M2::FromBits()))).c_str());
        case op::DataType::DT_FLOAT6_E2M3:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float6E2M3(v.ui8, op::Float6E2M3::FromBits()))).c_str());
        case op::DataType::DT_FLOAT4_E2M1:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float4E2M1(v.ui8, op::Float4E2M1::FromBits()))).c_str());
        case op::DataType::DT_FLOAT4_E1M2:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::Float4E1M2(v.ui8, op::Float4E1M2::FromBits()))).c_str());
        case op::DataType::DT_HIFLOAT8:
            return ge::AscendString(
                std::to_string(static_cast<float>(op::HiFloat8(v.ui8, op::HiFloat8::FromBits()))).c_str());
        case op::DataType::DT_INT8:
            return ge::AscendString(std::to_string(v.i8).c_str());
        case op::DataType::DT_INT16:
            return ge::AscendString(std::to_string(v.i16).c_str());
        case op::DataType::DT_UINT16:
            return ge::AscendString(std::to_string(v.ui16).c_str());
        case op::DataType::DT_UINT8:
            return ge::AscendString(std::to_string(v.ui8).c_str());
        case op::DataType::DT_INT32:
            return ge::AscendString(std::to_string(v.i32).c_str());
        case op::DataType::DT_INT64:
            return ge::AscendString(std::to_string(v.i64).c_str());
        case op::DataType::DT_UINT32:
            return ge::AscendString(std::to_string(v.ui32).c_str());
        case op::DataType::DT_UINT64:
            return ge::AscendString(std::to_string(v.ui64).c_str());
        case op::DataType::DT_BOOL:
            return ge::AscendString(std::to_string(v.i8).c_str());
        case op::DataType::DT_DOUBLE:
            return ge::AscendString(std::to_string(v.d).c_str());
        case op::DataType::DT_COMPLEX64: {
            std::ostringstream oss;
            oss << v.complex64;
            return ge::AscendString(oss.str().c_str());
        }
        case op::DataType::DT_COMPLEX128: {
            std::ostringstream oss;
            oss << v.complex128;
            return ge::AscendString(oss.str().c_str());
        }
        case op::DataType::DT_STRING:
        case op::DataType::DT_DUAL_SUB_INT8:
        case op::DataType::DT_DUAL_SUB_UINT8:
        case op::DataType::DT_QINT8:
        case op::DataType::DT_QINT16:
        case op::DataType::DT_QINT32:
        case op::DataType::DT_QUINT8:
        case op::DataType::DT_QUINT16:
        case op::DataType::DT_RESOURCE:
        case op::DataType::DT_STRING_REF:
        case op::DataType::DT_DUAL:
        case op::DataType::DT_VARIANT:
        case op::DataType::DT_UNDEFINED:
        case op::DataType::DT_INT4:
        case op::DataType::DT_UINT1:
        case op::DataType::DT_INT2:
        case op::DataType::DT_UINT2:
        default:
            OP_LOGW("no supported data type[%s].", op::ToString(dataType_).GetString());
            return "Unknown";
    }
}

float aclScalar::ToFloat() const { return To<float>(); }
double aclScalar::ToDouble() const { return To<double>(); }
bool aclScalar::ToBool() const { return To<bool>(); }
int8_t aclScalar::ToInt8() const { return To<int8_t>(); }
int16_t aclScalar::ToInt16() const { return To<int16_t>(); }
int32_t aclScalar::ToInt32() const { return To<int32_t>(); }
int64_t aclScalar::ToInt64() const { return To<int64_t>(); }
uint8_t aclScalar::ToUint8() const { return To<uint8_t>(); }
uint16_t aclScalar::ToUint16() const { return To<int16_t>(); }
uint32_t aclScalar::ToUint32() const { return To<uint32_t>(); }
uint64_t aclScalar::ToUint64() const { return To<uint64_t>(); }
op::fp16_t aclScalar::ToFp16() const { return To<op::fp16_t>(); }

op::bfloat16 aclScalar::ToBf16() const { return static_cast<op::bfloat16>(To<float>()); }

std::complex<float> aclScalar::ToComplex64() const { return To<std::complex<float>>(); }
std::complex<double> aclScalar::ToComplex128() const { return To<std::complex<double>>(); }

op::Float8E5M2 aclScalar::ToFloat8E5M2() const { return To<op::Float8E5M2>(); }

op::Float8E4M3FN aclScalar::ToFloat8E4M3FN() const { return To<op::Float8E4M3FN>(); }

op::Float8E8M0 aclScalar::ToFloat8E8M0() const { return To<op::Float8E8M0>(); }

op::Float6E3M2 aclScalar::ToFloat6E3M2() const { return To<op::Float6E3M2>(); }

op::Float6E2M3 aclScalar::ToFloat6E2M3() const { return To<op::Float6E2M3>(); }

op::Float4E2M1 aclScalar::ToFloat4E2M1() const { return To<op::Float4E2M1>(); }

op::Float4E1M2 aclScalar::ToFloat4E1M2() const { return To<op::Float4E1M2>(); }

op::HiFloat4 aclScalar::ToHiFloat4() const { return To<op::HiFloat4>(); }

op::HiFloat8 aclScalar::ToHiFloat8() const { return To<op::HiFloat8>(); }

aclScalar::aclScalar(const void* data, op::DataType dataType)
{
    dataType_ = dataType;
    switch (dataType_) {
        case op::DataType::DT_FLOAT:
            v.f = *static_cast<const float*>(data);
            break;
        case op::DataType::DT_FLOAT16:
            v.ui16 = *static_cast<const uint16_t*>(data);
            break;
        case op::DataType::DT_BF16:
            v.ui16 = *static_cast<const uint16_t*>(data);
            break;
        case op::DataType::DT_INT8:
            v.i8 = *static_cast<const int8_t*>(data);
            break;
        case op::DataType::DT_INT16:
            v.i16 = *static_cast<const int16_t*>(data);
            break;
        case op::DataType::DT_UINT16:
            v.ui16 = *static_cast<const uint16_t*>(data);
            break;
        case op::DataType::DT_UINT8:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_INT32:
            v.i32 = *static_cast<const int32_t*>(data);
            break;
        case op::DataType::DT_INT64:
            v.i64 = *static_cast<const int64_t*>(data);
            break;
        case op::DataType::DT_UINT32:
            v.ui32 = *static_cast<const uint32_t*>(data);
            break;
        case op::DataType::DT_UINT64:
            v.ui64 = *static_cast<const uint64_t*>(data);
            break;
        case op::DataType::DT_BOOL:
            v.i8 = *static_cast<const int8_t*>(data);
            break;
        case op::DataType::DT_DOUBLE:
            v.d = *static_cast<const double*>(data);
            break;
        case op::DataType::DT_COMPLEX64:
            v.complex64 = *static_cast<const std::complex<float>*>(data);
            break;
        case op::DataType::DT_COMPLEX128:
            v.complex128 = *static_cast<const std::complex<double>*>(data);
            break;
        case op::DataType::DT_FLOAT8_E5M2:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT8_E4M3FN:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT8_E8M0:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT6_E3M2:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT6_E2M3:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT4_E2M1:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_FLOAT4_E1M2:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        case op::DataType::DT_HIFLOAT8:
            v.ui8 = *static_cast<const uint8_t*>(data);
            break;
        default:
            std::string dtypeRange = "[DT_FLOAT(0), DT_FLOAT16(1), DT_INT8(2), DT_INT32(3), DT_UINT8(4), "
                                     "DT_INT16(6), DT_UINT16(7), DT_UINT32(8), DT_INT64(9), DT_UINT64(10), "
                                     "DT_DOUBLE(11), DT_BOOL(12), DT_COMPLEX64(16), DT_COMPLEX128(17), DT_BF16(27), "
                                     "DT_HIFLOAT8(34), DT_FLOAT8_E5M2(35), DT_FLOAT8_E4M3FN(36), DT_FLOAT8_E8M0(37), "
                                     "DT_FLOAT6_E3M2(38), DT_FLOAT6_E2M3(39), DT_FLOAT4_E2M1(40), DT_FLOAT4_E1M2(41)]";
            std::string dataTypeStr = std::string(op::ToString(dataType_).GetString()) + "(" +
                                      std::to_string(static_cast<int32_t>(dataType_)) + ")";
            OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(dataTypeStr.c_str(), dtypeRange.c_str());
            throw std::invalid_argument("dataType not supported.");
            break;
    }
}

aclScalar::aclScalar(int32_t value)
{
    dataType_ = op::DataType::DT_INT32;
    v.i32 = value;
}

aclScalar::aclScalar(int64_t value)
{
    dataType_ = op::DataType::DT_INT64;
    v.i64 = value;
}

aclScalar::aclScalar(int16_t value)
{
    dataType_ = op::DataType::DT_INT16;
    v.i16 = value;
}
aclScalar::aclScalar(int8_t value)
{
    dataType_ = op::DataType::DT_INT8;
    v.i8 = value;
}
aclScalar::aclScalar(uint32_t value)
{
    dataType_ = op::DataType::DT_UINT32;
    v.ui32 = value;
}

aclScalar::aclScalar(uint64_t value)
{
    dataType_ = op::DataType::DT_UINT64;
    v.ui64 = value;
}

aclScalar::aclScalar(uint16_t value)
{
    dataType_ = op::DataType::DT_UINT16;
    v.ui16 = value;
}
aclScalar::aclScalar(uint8_t value)
{
    dataType_ = op::DataType::DT_UINT8;
    v.ui8 = value;
}

aclScalar::aclScalar(float value)
{
    dataType_ = op::DataType::DT_FLOAT;
    v.f = value;
}

aclScalar::aclScalar(double value)
{
    dataType_ = op::DataType::DT_DOUBLE;
    v.d = value;
}

aclScalar::aclScalar(op::fp16_t value)
{
    dataType_ = op::DataType::DT_FLOAT;
    v.f = value;
}
aclScalar::aclScalar(op::bfloat16 value)
{
    dataType_ = op::DataType::DT_BF16;
    v.f = value;
}

aclScalar::aclScalar(op::Float8E5M2 value) : dataType_(op::DataType::DT_FLOAT8_E5M2) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float8E4M3FN value) : dataType_(op::DataType::DT_FLOAT8_E4M3FN) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float8E8M0 value) : dataType_(op::DataType::DT_FLOAT8_E8M0) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float6E3M2 value) : dataType_(op::DataType::DT_FLOAT6_E3M2) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float6E2M3 value) : dataType_(op::DataType::DT_FLOAT6_E2M3) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float4E2M1 value) : dataType_(op::DataType::DT_FLOAT4_E2M1) { v.ui8 = value.value; }

aclScalar::aclScalar(op::Float4E1M2 value) : dataType_(op::DataType::DT_FLOAT4_E1M2) { v.ui8 = value.value; }

aclScalar::aclScalar(op::HiFloat4 value) : dataType_(op::DataType::DT_FLOAT4_E1M2)
{
    v.ui8 = value.value;
    OP_LOGW("This is a reserved function for future, aclScalar of hifloat4 type cannot be created currently.");
}

aclScalar::aclScalar(op::HiFloat8 value) : dataType_(op::DataType::DT_HIFLOAT8) { v.ui8 = value.value; }

aclScalar::aclScalar(bool value)
{
    dataType_ = op::DataType::DT_BOOL;
    v.i8 = value;
}

const void* aclScalar::GetData() const { return &(v.d); }

uint64_t aclScalar::Size() const { return op::TypeSize(dataType_); }

op::DataType aclScalar::GetDataType() const { return dataType_; }

ge::AscendString aclScalar::ToString() const
{
    return ge::AscendString((std::string("aclScalar(data_type = ") + op::ToString(dataType_).GetString() +
                             ", value = " + ToStr().GetString() + ")")
                                .c_str());
}

ge::AscendString ToString(aclDataType dataType)
{
    static std::map<aclDataType, std::string> dtype = {{ACL_DT_UNDEFINED, "undefined"},
                                                       {ACL_FLOAT, "float32"},
                                                       {ACL_FLOAT16, "float16"},
                                                       {ACL_INT8, "int8"},
                                                       {ACL_INT32, "int32"},
                                                       {ACL_UINT8, "uint8"},
                                                       {ACL_INT16, "int16"},
                                                       {ACL_UINT16, "uint16"},
                                                       {ACL_UINT32, "uint32"},
                                                       {ACL_INT64, "int64"},
                                                       {ACL_UINT64, "uint64"},
                                                       {ACL_DOUBLE, "double"},
                                                       {ACL_BOOL, "bool"},
                                                       {ACL_STRING, "string"},
                                                       {ACL_COMPLEX64, "complex64"},
                                                       {ACL_COMPLEX128, "complex128"},
                                                       {ACL_BF16, "bfloat16"},
                                                       {ACL_HIFLOAT8, "hifloat8"},
                                                       {ACL_FLOAT8_E5M2, "float8_e5m2"},
                                                       {ACL_FLOAT8_E4M3FN, "float8_e4m3fn"},
                                                       {ACL_FLOAT8_E8M0, "float8_e8m0"},
                                                       {ACL_FLOAT6_E3M2, "float6_e3m2"},
                                                       {ACL_FLOAT6_E2M3, "float6_e2m3"},
                                                       {ACL_FLOAT4_E2M1, "float4_e2m1"},
                                                       {ACL_FLOAT4_E1M2, "float4_e1m2"}};
    auto iter = dtype.find(dataType);
    if (iter != dtype.end()) {
        return ge::AscendString((iter->second).c_str());
    } else {
        return "Unknown aclDataType";
    }
}

template bool aclScalar::CheckOverflows<bool>() const;
template bool aclScalar::CheckOverflows<int8_t>() const;
template bool aclScalar::CheckOverflows<uint8_t>() const;
template bool aclScalar::CheckOverflows<int16_t>() const;
template bool aclScalar::CheckOverflows<uint16_t>() const;
template bool aclScalar::CheckOverflows<int32_t>() const;
template bool aclScalar::CheckOverflows<uint32_t>() const;
template bool aclScalar::CheckOverflows<int64_t>() const;
template bool aclScalar::CheckOverflows<uint64_t>() const;
template bool aclScalar::CheckOverflows<op::fp16_t>() const;
template bool aclScalar::CheckOverflows<op::bfloat16>() const;
template bool aclScalar::CheckOverflows<float>() const;
template bool aclScalar::CheckOverflows<double>() const;
template bool aclScalar::CheckOverflows<std::complex<float>>() const;
template bool aclScalar::CheckOverflows<std::complex<double>>() const;
template bool aclScalar::CheckOverflows<op::Float8E5M2>() const;
template bool aclScalar::CheckOverflows<op::Float8E4M3FN>() const;
template bool aclScalar::CheckOverflows<op::Float8E8M0>() const;
template bool aclScalar::CheckOverflows<op::Float6E3M2>() const;
template bool aclScalar::CheckOverflows<op::Float6E2M3>() const;
template bool aclScalar::CheckOverflows<op::Float4E2M1>() const;
template bool aclScalar::CheckOverflows<op::Float4E1M2>() const;
template bool aclScalar::CheckOverflows<op::HiFloat4>() const;
template bool aclScalar::CheckOverflows<op::HiFloat8>() const;

ge::AscendString ToString(const aclTensor* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclTensorList* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclIntArray* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclBoolArray* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclFloatArray* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclFp16Array* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclBf16Array* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclScalar* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}
ge::AscendString ToString(const aclScalarList* t)
{
    if (t == nullptr) {
        return "nullptr";
    } else {
        return t->ToString();
    }
}

ge::AscendString ToString(const aclTensor& t) { return t.ToString(); }

ge::AscendString ToString(const aclTensorList& t) { return t.ToString(); }

ge::AscendString ToString(const aclIntArray& t) { return t.ToString(); }

ge::AscendString ToString(const aclBoolArray& t) { return t.ToString(); }

ge::AscendString ToString(const aclFloatArray& t) { return t.ToString(); }

ge::AscendString ToString(const aclFp16Array& t) { return t.ToString(); }

ge::AscendString ToString(const aclBf16Array& t) { return t.ToString(); }

ge::AscendString ToString(const aclScalar& t) { return t.ToString(); }

ge::AscendString ToString(const aclScalarList& t) { return t.ToString(); }

template <typename T>
aclTensor::aclTensor(const T* value, uint64_t size, op::DataType dataType)
    : viewOffset_(0),
      viewStrides_(),
      viewShape_({static_cast<int64_t>(size)}),
      viewFormat_(op::Format::FORMAT_ND),
      isView_(false)
{
    auto typeSize = op::TypeSize(dataType);
    void* dataAddr = op::internal::Allocate(size * typeSize);
    OP_CHECK(dataAddr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    storage_ = new (std::nothrow) aclStorage(dataAddr, false);
    OP_CHECK(storage_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    op::StorageShape gert_shape({static_cast<int64_t>(size)}, {static_cast<int64_t>(size)});
    op::StorageFormat gert_format;
    gert_format.SetStorageFormat(op::Format::FORMAT_ND);
    gert_format.SetOriginFormat(op::Format::FORMAT_ND);
    op::ToContiguousStrides(viewShape_, viewStrides_);
    void* base = op::internal::Allocate(sizeof(op::Tensor));
    OP_CHECK(base != nullptr, OP_LOGE(ACLNN_ERR_INNER, "aclTensor allocate failed."), throw std::bad_alloc());
    tensor_ = new (base) op::Tensor(gert_shape, gert_format, op::TensorPlacement::kOnHost, dataType, dataAddr);
    switch (dataType) {
        case op::DataType::DT_FLOAT: {
            auto tmpDataAddr = static_cast<float*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<float>(value[i]);
            }
            break;
        }
        case op::DataType::DT_FLOAT16: {
            auto tmpDataAddr = static_cast<op::fp16_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<op::fp16_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_BF16: {
            auto tmpDataAddr = static_cast<op::bfloat16*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<op::bfloat16>(value[i]);
            }
            break;
        }
        case op::DataType::DT_INT8: {
            auto tmpDataAddr = static_cast<int8_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<int8_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_INT16: {
            auto tmpDataAddr = static_cast<int16_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<int16_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_UINT16: {
            auto tmpDataAddr = static_cast<uint16_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<uint16_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_UINT8: {
            auto tmpDataAddr = static_cast<uint8_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<uint8_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_INT32: {
            auto tmpDataAddr = static_cast<int32_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<int32_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_INT64: {
            auto tmpDataAddr = static_cast<int64_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<int64_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_UINT32: {
            auto tmpDataAddr = static_cast<uint32_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<uint32_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_UINT64: {
            auto tmpDataAddr = static_cast<uint64_t*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<uint64_t>(value[i]);
            }
            break;
        }
        case op::DataType::DT_BOOL: {
            auto tmpDataAddr = static_cast<bool*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                if constexpr (std::is_floating_point<T>::value ||
                              std::is_same<op::fp16_t, typename std::decay<T>::type>::value) {
                    *(tmpDataAddr + i) = std::abs(value[i]) >= std::numeric_limits<float>::epsilon();
                } else {
                    *(tmpDataAddr + i) = static_cast<bool>(value[i]);
                }
            }
            break;
        }
        case op::DataType::DT_DOUBLE: {
            auto tmpDataAddr = static_cast<double*>(dataAddr);
            for (uint64_t i = 0; i < size; i++) {
                *(tmpDataAddr + i) = static_cast<double>(value[i]);
            }
            break;
        }
        default:
            std::string dtypeRange = "[DT_FLOAT(0), DT_FLOAT16(1), DT_INT8(2), DT_INT32(3), DT_UINT8(4), "
                                     "DT_INT16(6), DT_UINT16(7), DT_UINT32(8), DT_INT64(9), DT_UINT64(10), "
                                     "DT_DOUBLE(11), DT_BOOL(12), DT_BF16(27)]";
            std::string dataTypeStr = std::string(op::ToString(dataType).GetString()) + "(" +
                                      std::to_string(static_cast<int32_t>(dataType)) + ")";
            OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(dataTypeStr.c_str(), dtypeRange.c_str());
            throw std::invalid_argument("dataType not supported.");
            break;
    }
}

template class aclArray<int64_t>;
template class aclArray<float>;
template class aclArray<bool>;
template class aclArray<op::fp16_t>;
template class aclArray<op::bfloat16>;

template aclTensor::aclTensor(const int64_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const uint64_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const int32_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const uint32_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const int8_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const uint8_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const int16_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const uint16_t* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const float* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const double* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const bool* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const char* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const op::bfloat16* value, uint64_t size, op::DataType dataType);
template aclTensor::aclTensor(const op::fp16_t* value, uint64_t size, op::DataType dataType);
