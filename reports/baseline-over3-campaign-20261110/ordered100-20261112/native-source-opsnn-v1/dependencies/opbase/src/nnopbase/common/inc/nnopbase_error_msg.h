/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_NNOPBASE_ERROR_MSG_H
#define OP_API_OP_API_COMMON_INC_NNOPBASE_ERROR_MSG_H

#include "base/err_msg.h"
#include "opdev/op_log.h"

// ============================================================================
// ERROR MSG FOR NNOPBASE (EZ1002-EZ1009)
// ============================================================================
#define OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE(funcDesc, envName)                                      \
    do {                                                                                                              \
        std::string msg = std::string(funcDesc) + " failed. Environment variable " + envName + " is not configured."; \
        const std::vector<const char*> msgKey = {"funcDesc", "envName"};                                              \
        const std::vector<const char*> msgValue = {funcDesc, envName};                                                \
        OP_LOGE_WITHOUT_REPORT("EZ1002", "%s", msg.c_str());                                                          \
        REPORT_PREDEFINED_ERR_MSG("EZ1002", msgKey, msgValue);                                                        \
    } while (false)

#define OP_LOGE_FOR_FILE_OPERATION_ERROR_OPEN(file, reason)                                         \
    do {                                                                                            \
        std::string msg = std::string("Failed to open file ") + file + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"file", "reason"};                                 \
        const std::vector<const char*> msgValue = {file, reason};                                   \
        OP_LOGE_WITHOUT_REPORT("EZ1003", "%s", msg.c_str());                                        \
        REPORT_PREDEFINED_ERR_MSG("EZ1003", msgKey, msgValue);                                      \
    } while (false)

#define OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(file, reason)                                         \
    do {                                                                                             \
        std::string msg = std::string("Failed to parse file ") + file + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"file", "reason"};                                  \
        const std::vector<const char*> msgValue = {file, reason};                                    \
        OP_LOGE_WITHOUT_REPORT("EZ1004", "%s", msg.c_str());                                         \
        REPORT_PREDEFINED_ERR_MSG("EZ1004", msgKey, msgValue);                                       \
    } while (false)

#define OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE_WITH_INVALID_CONTENT(file, reason)                    \
    do {                                                                                             \
        std::string msg = std::string("Failed to parse file ") + file + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"file", "reason"};                                  \
        const std::vector<const char*> msgValue = {file, reason};                                    \
        OP_LOGE_WITHOUT_REPORT("EZ1005", "%s", msg.c_str());                                         \
        REPORT_PREDEFINED_ERR_MSG("EZ1005", msgKey, msgValue);                                       \
    } while (false)

#define OP_LOGE_FOR_NOT_SUPPORTED_DATA_TYPE(dataType, dtypeRange)                            \
    do {                                                                                     \
        std::string opName = op::internal::GetLogApiInfo();                                  \
        std::string msg = "Operator " + opName + " does not support data type " + dataType + \
                          ". The supported data type range is " + dtypeRange + ".";          \
        const std::vector<const char*> msgKey = {"opName", "dataType", "dtypeRange"};        \
        const std::vector<const char*> msgValue = {opName.c_str(), dataType, dtypeRange};    \
        OP_LOGE_WITHOUT_REPORT("EZ1006", "%s", msg.c_str());                                 \
        REPORT_PREDEFINED_ERR_MSG("EZ1006", msgKey, msgValue);                               \
    } while (false)

#define OP_LOGE_FOR_INVALID_ARGUMENT_TENSOR_INPUT_SHAPE(paraName, dim, reason)                        \
    do {                                                                                              \
        std::string opName = op::internal::GetLogApiInfo();                                           \
        std::string msg = std::string("Shape ") + paraName + " of the tensor of operator " + opName + \
                          " has incorrect dimension " + dim + ". Reason: " + reason + ".";            \
        const std::vector<const char*> msgKey = {"paraName", "opName", "dim", "reason"};              \
        const std::vector<const char*> msgValue = {paraName, opName.c_str(), dim, reason};            \
        OP_LOGE_WITHOUT_REPORT("EZ1007", "%s", msg.c_str());                                          \
        REPORT_PREDEFINED_ERR_MSG("EZ1007", msgKey, msgValue);                                        \
    } while (false)

#define OP_LOGE_FOR_EXECUTION_TILING_ERROR(reason)                                              \
    do {                                                                                        \
        std::string opName = op::internal::GetLogApiInfo();                                     \
        std::string msg = "Failed to execute operator " + opName + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"opName", "reason"};                           \
        const std::vector<const char*> msgValue = {opName.c_str(), reason};                     \
        OP_LOGE_WITHOUT_REPORT("EZ1008", "%s", msg.c_str());                                    \
        REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);                                  \
    } while (false)

#define OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(reason)                                    \
    do {                                                                                        \
        std::string opName = op::internal::GetLogApiInfo();                                     \
        std::string msg = "Failed to execute operator " + opName + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"opName", "reason"};                           \
        const std::vector<const char*> msgValue = {opName.c_str(), reason};                     \
        OP_LOGE_WITHOUT_REPORT("EZ1009", "%s", msg.c_str());                                    \
        REPORT_PREDEFINED_ERR_MSG("EZ1009", msgKey, msgValue);                                  \
    } while (false)

#define OP_LOGE_FOR_INVALID_ARGUMENT_WITHOUT_SOLUTION(value, paraName, reason)           \
    do {                                                                                 \
        std::string msg = std::string("Value ") + value + " for parameter " + paraName + \
                          " is invalid. Reason: " + reason + ".";                        \
        const std::vector<const char*> msgKey = {"value", "paraName", "reason"};         \
        const std::vector<const char*> msgValue = {value, paraName, reason};             \
        OP_LOGE_WITHOUT_REPORT("EZ1010", "%s", msg.c_str());                             \
        REPORT_PREDEFINED_ERR_MSG("EZ1010", msgKey, msgValue);                           \
    } while (false)

#define OP_LOGE_FOR_INVALID_ARGUMENT_NULL_POINTER(functionName, paraName)                                           \
    do {                                                                                                            \
        std::string msg = std::string(functionName) + " failed because " + paraName + " cannot be a NULL pointer."; \
        const std::vector<const char*> msgKey = {"functionName", "paraName"};                                       \
        const std::vector<const char*> msgValue = {functionName, paraName};                                         \
        OP_LOGE_WITHOUT_REPORT("EZ1011", "%s", msg.c_str());                                                        \
        REPORT_PREDEFINED_ERR_MSG("EZ1011", msgKey, msgValue);                                                      \
    } while (false)

#define NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(x, ret)                  \
    do {                                                             \
        if ((x) == nullptr) {                                        \
            OP_LOGE_FOR_INVALID_ARGUMENT_NULL_POINTER(__func__, #x); \
            return (ret);                                            \
        }                                                            \
    } while (false)

#define OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(value, paraName, maxValue)         \
    do {                                                                                   \
        std::string msg = std::string("Value ") + value + " for parameter " + paraName +   \
                          " is out of range, it should be smaller than " + maxValue + "."; \
        const std::vector<const char*> msgKey = {"value", "paraName", "maxValue"};         \
        const std::vector<const char*> msgValue = {value, paraName, maxValue};             \
        OP_LOGE_WITHOUT_REPORT("EZ1012", "%s", msg.c_str());                               \
        REPORT_PREDEFINED_ERR_MSG("EZ1012", msgKey, msgValue);                             \
    } while (false)

#define NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(value, paraName, maxValue)                                         \
    do {                                                                                                      \
        if ((value) >= (maxValue)) {                                                                          \
            std::string valueStr = std::to_string(value);                                                     \
            std::string maxValueStr = std::to_string(maxValue);                                               \
            OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(valueStr.c_str(), paraName, maxValueStr.c_str()); \
            return ACLNN_ERR_PARAM_INVALID;                                                                   \
        }                                                                                                     \
    } while (false)

#define OP_LOGE_FOR_CONFIG_FILE_NOT_EXIST_IN_DYNAMIC_SHAPE()                                                   \
    do {                                                                                                       \
        std::string opName = op::internal::GetLogApiInfo();                                                    \
        std::string msg = "In the dynamic shape scenario, the JSON configuration file of operator " + opName + \
                          " cannot be found.";                                                                 \
        const std::vector<const char*> msgKey = {"opName"};                                                    \
        const std::vector<const char*> msgValue = {opName.c_str()};                                            \
        OP_LOGE_WITHOUT_REPORT("EZ1013", "%s", msg.c_str());                                                   \
        REPORT_PREDEFINED_ERR_MSG("EZ1013", msgKey, msgValue);                                                 \
    } while (false)

#define OP_LOGE_FOR_EXECUTION_INFERSHAPE_ERROR(reason)                                          \
    do {                                                                                        \
        std::string opName = op::internal::GetLogApiInfo();                                     \
        std::string msg = "Failed to execute operator " + opName + ". Reason: " + reason + "."; \
        const std::vector<const char*> msgKey = {"opName", "reason"};                           \
        const std::vector<const char*> msgValue = {opName.c_str(), reason};                     \
        OP_LOGE_WITHOUT_REPORT("EZ1014", "%s", msg.c_str());                                    \
        REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);                                  \
    } while (false)
#endif // OP_API_OP_API_COMMON_INC_NNOPBASE_ERROR_MSG_H