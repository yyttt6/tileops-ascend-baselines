/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>
#include <graph/utils/type_utils.h>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"

namespace op {

constexpr int64_t BIT_TO_BYTE_SHIFT = 3;
constexpr int64_t BIT_TO_BTYE_ALIGN = 7;

using std::map;
using std::string;

DataType ToDataType(const string& dataTypeStr) { return ge::TypeUtils::SerialStringToDataType(dataTypeStr); }

ge::AscendString ToString(DataType dataType)
{
    return ge::AscendString(ge::TypeUtils::DataTypeToSerialString(dataType).c_str());
}

ge::AscendString ToString(const std::initializer_list<DataType>& dataTypes)
{
    if (!std::empty(dataTypes)) {
        std::string str("[");
        for (auto type : dataTypes) {
            str += ToString(type).GetString();
            str += ",";
        }
        str += "]";
        return ge::AscendString(str.c_str());
    }

    return ge::AscendString("[]");
}

int64_t CalcShapeBytes(int64_t size, DataType dataType, bool ceil)
{
    int64_t byteNum = 0;
    size_t typeSize = op::TypeSize(dataType);
    if (typeSize > kDataTypeSizeBitOffset) {
        if (ge::MulOverflow((typeSize - kDataTypeSizeBitOffset), size, byteNum)) {
            OP_LOGW("Calc shape bytes overflow, shape : %ld, type size : %zu.", size, typeSize);
        }
        // convert bit to Byte
        if (!ceil) {
            byteNum = (byteNum + BIT_TO_BTYE_ALIGN) >> BIT_TO_BYTE_SHIFT;
        } else {
            byteNum = byteNum >> BIT_TO_BYTE_SHIFT;
        }
    } else {
        if (ge::MulOverflow(typeSize, size, byteNum)) {
            OP_LOGW("Calc shape bytes overflow, shape : %ld, type size : %zu.", size, typeSize);
        }
    }
    return byteNum;
}

} // namespace op
