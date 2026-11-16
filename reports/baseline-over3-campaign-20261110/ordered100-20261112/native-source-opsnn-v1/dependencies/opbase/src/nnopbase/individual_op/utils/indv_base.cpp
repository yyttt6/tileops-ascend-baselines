/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_base.h"
#include <string>
#include <map>
#include <functional>
#include "opdev/small_vector.h"
#include "opdev/op_dfx.h"

namespace nnopbase {
namespace {
using namespace op::internal;

const std::map<NnopbaseAttrDtype, std::string> kDTypeToStr = {
    {kNnopbaseBool, "Bool"}, {kNnopbaseFloat, "Float"}, {kNnopbaseInt, "Int"}, {kNnopbaseString, "String"}};

template <typename T>
std::string DataToStr(const void* data)
{
    const T* p = op::internal::PtrCastTo<const T>(data);
    if constexpr (std::is_same<char, T>::value) {
        return std::string(p);
    } else {
        return std::to_string(*p);
    }
}

const std::map<NnopbaseAttrDtype, std::function<std::string(const void*)>> kPrintFunc = {
    {kNnopbaseBool, DataToStr<bool>},
    {kNnopbaseFloat, DataToStr<float>},
    {kNnopbaseInt, DataToStr<int64_t>},
    {kNnopbaseString, DataToStr<char>}};

template <typename T>
std::string ToString(const T* data)
{
    if constexpr (std::is_same<op::fp16_t, typename std::decay<T>::type>::value) {
        return std::to_string(static_cast<T>(data[0]).toFloat());
    } else if constexpr (std::is_same<op::bfloat16, typename std::decay<T>::type>::value) {
        return std::to_string(float(*data));
    } else {
        return std::to_string(*data);
    }
}

template <typename T>
std::string ConvertToStr(const GertTensor& tensor)
{
    const ge::DataType dataType = tensor.GetDataType();
    const size_t len = static_cast<size_t>(tensor.GetStorageShape().GetShapeSize());
    const int32_t dtypeSize = GetSizeByDataType(dataType);
    const void* addr = tensor.GetAddr();
    std::string res;
    for (size_t i = 0U; i < len; i++) {
        res += ToString(op::internal::PtrCastTo<const T>(op::internal::PtrCastTo<const uint8_t>(addr) + i * dtypeSize));
        if (i + 1 < len) { // 非最后一个
            // 逗号后不能加空格，fp32类型数据用sqlite解析的时候会受空格影响导致少数据
            res += ",";
        }
    }

    return res;
}

const std::map<ge::DataType, std::function<std::string(const GertTensor&)>> kCovertFunc = {
    {ge::DT_FLOAT, ConvertToStr<float>},      {ge::DT_FLOAT16, ConvertToStr<op::fp16_t>},
    {ge::DT_INT8, ConvertToStr<int8_t>},      {ge::DT_INT32, ConvertToStr<int32_t>},
    {ge::DT_UINT8, ConvertToStr<uint8_t>},    {ge::DT_INT16, ConvertToStr<int16_t>},
    {ge::DT_UINT16, ConvertToStr<uint16_t>},  {ge::DT_UINT32, ConvertToStr<uint32_t>},
    {ge::DT_INT64, ConvertToStr<int64_t>},    {ge::DT_UINT64, ConvertToStr<uint64_t>},
    {ge::DT_DOUBLE, ConvertToStr<double>},    {ge::DT_BOOL, ConvertToStr<bool>},
    {ge::DT_BF16, ConvertToStr<op::bfloat16>}};
} // namespace

std::string ToStr(const GertTensor& tensor)
{
    const ge::DataType dataType = tensor.GetDataType();
    const auto& it = kCovertFunc.find(dataType);
    if (it != kCovertFunc.end()) {
        return (it->second)(tensor);
    } else {
        OP_LOGW("Tensor dtype %s cannot be reported.", op::ToString(dataType).GetString());
        std::stringstream res;
        NnopbasePrintHex(PtrCastTo<uint8_t>(tensor.GetAddr()), tensor.GetSize(), res);
        std::string str = res.str();
        if (!str.empty() && str.back() == ' ') {
            str.pop_back();
        }
        return str;
    }
}

std::string ToStr(const NnopbaseAttr& attr)
{
    const auto& it = kPrintFunc.find(attr.dtype);
    if (it == kPrintFunc.end()) {
        // 维测能力，对于不支持的类型打印16进制字节码
        std::stringstream attrStr;
        NnopbasePrintHex(PtrCastTo<uint8_t>(attr.addr.addr), attr.addr.size, attrStr);
        return attrStr.str();
    }
    if (attr.addr.isVector) {
        std::string res;
        const size_t len = attr.addr.size / attr.addr.elementSize;
        for (size_t i = 0U; i < len; ++i) {
            void* addr = ValueToPtr(PtrToValue(attr.addr.addr) + i * attr.addr.elementSize);
            res += (it->second)(addr);
            if (i + 1 < len) { // 非最后一个
                               // 逗号后不能加空格，fp32类型数据用sqlite解析的时候会受空格影响导致少数据
                res += ",";
            }
        }
        return res;
    } else {
        return attr.addr.addr == nullptr ? "" : (it->second)(attr.addr.addr);
    }
}

std::string ToStr(const NnopbaseAttrDtype dtype)
{
    const auto& it = kDTypeToStr.find(dtype);
    if (it != kDTypeToStr.end()) {
        return it->second;
    } else {
        // 维测能力，不打印error，将传入的dtype实际值打印出来
        return std::to_string(dtype);
    }
}

std::string ToStr(const NnopbaseAttrs& attrs)
{
    std::string res;
    for (size_t i = 0U; i < attrs.num; i++) {
        res += "[";
        res += ToStr(attrs.attrs[i]);
        res += "]";
        if (i + 1 < attrs.num) { // 非最后一个
            res += ", ";
        }
    }
    return res;
}
} // namespace nnopbase
