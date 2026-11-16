/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_any_value.h"
#include "graph/types.h"
#include "opdev/op_log.h"

namespace op {
namespace internal {
constexpr int32_t TypeIdBoolValue = 1;
constexpr int32_t TypeIdStringValue = 2;
constexpr int32_t TypeIdFloatValue = 3;
constexpr int32_t TypeIdInt64Value = 4;
constexpr int32_t TypeIdVectorVectorInt64Value = 10;
constexpr int32_t TypeIdDataTypeValue = 11;
constexpr int32_t TypeIdVectorVectorFloatValue = 12;
constexpr int32_t TypeIdVectorStringValue = 13;
constexpr int32_t TypeIdVectorFloatValue = 14;
constexpr int32_t TypeIdVectorBoolValue = 15;
constexpr int32_t TypeIdVectorInt64Value = 16;
constexpr int32_t TypeIdVectorDataTypeValue = 22;

template <>
TypeId GetTypeId<bool>()
{
    return reinterpret_cast<TypeId>(TypeIdBoolValue);
}

template <>
TypeId GetTypeId<std::string>()
{
    return reinterpret_cast<TypeId>(TypeIdStringValue);
}

template <>
TypeId GetTypeId<float>()
{
    return reinterpret_cast<TypeId>(TypeIdFloatValue);
}

template <>
TypeId GetTypeId<int64_t>()
{
    return reinterpret_cast<TypeId>(TypeIdInt64Value);
}

template <>
TypeId GetTypeId<std::vector<std::vector<int64_t>>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorVectorInt64Value);
}

template <>
TypeId GetTypeId<ge::DataType>()
{
    return reinterpret_cast<TypeId>(TypeIdDataTypeValue);
}

template <>
TypeId GetTypeId<std::vector<std::vector<float>>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorVectorFloatValue);
}

template <>
TypeId GetTypeId<std::vector<std::string>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorStringValue);
}

template <>
TypeId GetTypeId<std::vector<float>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorFloatValue);
}

template <>
TypeId GetTypeId<std::vector<bool>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorBoolValue);
}

template <>
TypeId GetTypeId<std::vector<int64_t>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorInt64Value);
}

template <>
TypeId GetTypeId<std::vector<ge::DataType>>()
{
    return reinterpret_cast<TypeId>(TypeIdVectorDataTypeValue);
}

namespace {
std::unordered_map<TypeId, AnyValue::ValueType> type_ids_to_value_type = {
    {nullptr, AnyValue::VT_NONE},
    {GetTypeId<std::string>(), AnyValue::VT_STRING},
    {GetTypeId<float>(), AnyValue::VT_FLOAT},
    {GetTypeId<bool>(), AnyValue::VT_BOOL},
    {GetTypeId<int64_t>(), AnyValue::VT_INT},
    {GetTypeId<std::vector<std::vector<int64_t>>>(), AnyValue::VT_LIST_LIST_INT},
    {GetTypeId<ge::DataType>(), AnyValue::VT_DATA_TYPE},
    {GetTypeId<std::vector<std::vector<float>>>(), AnyValue::VT_LIST_LIST_FLOAT},
    {GetTypeId<std::vector<std::string>>(), AnyValue::VT_LIST_STRING},
    {GetTypeId<std::vector<float>>(), AnyValue::VT_LIST_FLOAT},
    {GetTypeId<std::vector<bool>>(), AnyValue::VT_LIST_BOOL},
    {GetTypeId<std::vector<int64_t>>(), AnyValue::VT_LIST_INT},
    {GetTypeId<std::vector<ge::DataType>>(), AnyValue::VT_LIST_DATA_TYPE},
};
} // namespace

AnyValue::AnyValue(AnyValue&& other) noexcept
{
    if (!other.IsEmpty()) {
        other.operate_(OperateType::kOpMove, &other, this);
    }
}

AnyValue& AnyValue::operator=(AnyValue&& other) noexcept
{
    if (&other == this) {
        return *this;
    }
    Clear();
    if (!other.IsEmpty()) {
        other.operate_(OperateType::kOpMove, &other, this);
    }
    return *this;
}

TypeId AnyValue::GetValueTypeId() const noexcept
{
    TypeId vt{kInvalidTypeId};
    if (!IsEmpty()) {
        operate_(OperateType::kGetTypeId, this, &vt);
    }
    return vt;
}

AnyValue::ValueType AnyValue::GetValueType() const noexcept
{
    auto vt = GetValueTypeId();
    auto iter = type_ids_to_value_type.find(vt);
    if (iter == type_ids_to_value_type.end()) {
        return AnyValue::VT_NONE;
    }
    return iter->second;
}

const void* AnyValue::GetAddr() const
{
    void* addr = nullptr;
    operate_(OperateType::kOpGetAddr, this, &addr);
    return addr;
}
} // namespace internal
} // namespace op