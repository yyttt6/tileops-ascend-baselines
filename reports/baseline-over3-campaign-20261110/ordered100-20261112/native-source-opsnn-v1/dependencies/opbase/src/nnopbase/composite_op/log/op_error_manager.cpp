/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"

#include <string>
#include <vector>
#include "securec.h"
#include "base/err_msg.h"
#include "base/dlog_pub.h"

static const std::string g_errorInfoJson = R"(
{
    "error_info_list": [
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Inner_Error",
        "ErrCode": "EZ9999",
        "ErrMessage": "%s",
        "Arglist": "message",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "1. Check whether the CANN package is correct.\n2. Check whether the environment variable is correct."
        }
    },
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Parameter_Error",
        "ErrCode": "EZ1001",
        "ErrMessage": "%s",
        "Arglist": "message"
    },
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Runtime_Error",
        "ErrCode": "EZ9903",
        "ErrMessage": "%s",
        "Arglist": "message",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "In this scenario, collect the plog when the fault occurs and locate the fault based on the plog."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Config_Error_Invalid_Environment_Variable",
        "ErrCode": "EZ1002",
        "ErrMessage": "%s failed. Environment variable %s is not configured.",
        "Arglist": "funcDesc, envName",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "Reset the environment variable by referring to the Environment Variable Reference."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Open",
        "ErrCode": "EZ1003",
        "ErrMessage": "Failed to open file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Parse",
        "ErrCode": "EZ1004",
        "ErrMessage": "Failed to parse file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Parse",
        "ErrCode": "EZ1005",
        "ErrMessage": "Failed to parse file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "1.The custom operator JSON file is damaged. 2.The built-in operator JSON file is damaged.",
            "Solution": "1.Reinstall the custom operator package. 2.Reinstall the built-in operator package."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Not_Supported_Data_Type",
        "ErrCode": "EZ1006",
        "ErrMessage": "Operator %s does not support data type %s. The supported data type range is %s.",
        "Arglist": "opName, dataType, dtypeRange",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Invalid_Argument_Tensor_Input_Shape",
        "ErrCode": "EZ1007",
        "ErrMessage": "Shape %s of the tensor of operator %s has incorrect dimension %s. Reason: %s.",
        "Arglist": "paraName, opName, dim, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Execution_Error",
        "ErrCode": "EZ1008",
        "ErrMessage": "Failed to execute operator %s. Reason: %s.",
        "Arglist": "opName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "1.If the tiling function does not exist, check whether the tiling function is registered successfully. 2.If the tiling function fails to be executed, check the implementation logic of the tiling function."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Execution_Error",
        "ErrCode": "EZ1009",
        "ErrMessage": "Failed to execute operator %s. Reason: %s.",
        "Arglist": "opName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Invalid_Argument",
        "ErrCode": "EZ1010",
        "ErrMessage": "Value %s for parameter %s is invalid. Reason: %s.",
        "Arglist": "value, paraName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Invalid_Argument_Null_Pointer",
        "ErrCode": "EZ1011",
        "ErrMessage": "%s failed because %s cannot be a NULL pointer.",
        "Arglist": "functionName, paraName",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Invalid_Argument",
        "ErrCode": "EZ1012",
        "ErrMessage": "Value %s for parameter %s is out of range, it should be smaller than %s.",
        "Arglist": "value, paraName, maxValue",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Config_Error",
        "ErrCode": "EZ1013",
        "ErrMessage": "In the dynamic shape scenario, the JSON configuration file of operator %s cannot be found.",
        "Arglist": "opName",
        "suggestion": {
            "Possible Cause": "1.The operator package of the corresponding SoC version is not correctly installed. 2.If the operator is a custom operator, the environment variable ASCEND_CUSTOM_OPP_PATH of the custom operator may not be configured or be incorrectly configured. 3.If the operator is a built-in operator, the environment variable ASCEND_OPP_PATH of the built-in operator may not be configured or be incorrectly configured.",
            "Solution": "1.Install the operator package of the corresponding SoC version correctly. 2.Set ASCEND_CUSTOM_OPP_PATH to the actual installation path of the custom operator package. 3.Set ASCEND_OPP_PATH to the actual installation path of the built-in operator package."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Execution_Error",
        "ErrCode": "EZ1014",
        "ErrMessage": "Failed to execute operator %s. Reason: %s.",
        "Arglist": "opName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "1.If the inferShape function does not exist, check whether the inferShape function is registered successfully. 2.If the inferShape function fails to be executed, check the implementation logic of the inferShape function."
        }
    }
    ]
})";

REG_FORMAT_ERROR_MSG(g_errorInfoJson.c_str(), g_errorInfoJson.length());

constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string& code, const char* fmt, ...)
{
    std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');

    va_list argList;
    va_start(argList, fmt);
    auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
    if (ret == -1) {
        OP_LOGW("Construct report error message fail, maybe the length of error message exceed limits: %zu",
                LIMIT_PREDEFINED_MESSAGE);
    }
    va_end(argList);

    const std::vector<const char*> msgKey = {"message"};
    const std::vector<const char*> msgvalue = {buf.data()};
    REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgvalue);
}

void DlogRecordInner(int32_t moduleId, int32_t level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    DlogVaList(moduleId, level, fmt, args);
    va_end(args);
}

int32_t CheckLogLevelInner(int32_t moduleId, int32_t level) { return CheckLogLevel(moduleId, level); }