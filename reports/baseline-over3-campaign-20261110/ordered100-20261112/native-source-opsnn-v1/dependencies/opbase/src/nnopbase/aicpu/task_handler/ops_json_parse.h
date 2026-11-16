/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_OPS_JSON_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_OPS_JSON_H_

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "ops_struct.h"
#include "aclnn/aclnn_base.h"

namespace {
// op kernel json file configuration item: op type
const std::string kConfigOpType = "opType";

// op kernel json file configuration item: engine information that the op
const std::string kConfigOpInfo = "opInfo";

// op kernel json file configuration item: which opsKernelStore
const std::string kConfigKernelLib = "opKernelLib";

// op kernel json file configuration item: whether user defined
const std::string kConfigUserDefined = "userDefined";

// op kernel json file configuration item: kernel so name
const std::string kConfigKernelSo = "kernelSo";

// op kernel json file configuration item: function name
const std::string kConfigFunctionName = "functionName";

// op kernel json file configuration item: all op infos
const std::string kConfigOpInfos = "opInfos";

// op kernel json file configuration item: kernel lib name
const std::string kConfigLibName = "libName";

} // namespace

namespace op {
namespace internal {
class OpsJsonFile {
public:
    /**
     * Get instance
     * @return OpsJsonFile instance reference
     */
    static OpsJsonFile& Instance();

    /**
     * Default Destructor
     */
    virtual ~OpsJsonFile() = default;

    /**
     * Read json file in specified path(based on source file's current path)
     * @param filePath json file path
     * @param jsonRead read json handle
     * @return whether read file successfully
     */
    aclnnStatus ParseUnderPath(const std::string& filePath, nlohmann::json& jsonRead) const;

    // Copy operations are prohibited
    OpsJsonFile(const OpsJsonFile&) = delete;
    // Copy operations are prohibited
    OpsJsonFile& operator=(const OpsJsonFile&) = delete;
    // Move operations are prohibited
    OpsJsonFile(OpsJsonFile&&) = delete;
    // Move operations are prohibited
    OpsJsonFile& operator=(OpsJsonFile&&) = delete;

private:
    // Default Constructor
    OpsJsonFile() = default;

    bool ConvertJsonFormat(nlohmann::json& jsonRead) const;
    aclnnStatus StringToBool(const std::string& str, bool& result) const;
    bool CheckAndGetUserDefine(const nlohmann::json& buff, const std::string& opType, const std::string& fieldStr,
                               bool& value) const;
};

/**
 * OpInfoDescs json to struct object function
 * @param jsonRead read json handle
 * @param infos all op infos
 * @return whether read file successfully
 */
void from_json(const nlohmann::json& jsonRead, OpInfoDescs& infos);

void from_json(const nlohmann::json& jsonRead, OpInfoDesc& desc);

/**
 * OpInfo json to struct object function
 * @param jsonRead read json handle
 * @param op_info engine information that the op
 * @return whether read file successfully
 */
void from_json(const nlohmann::json& jsonRead, OpFullInfo& opInfo);

} // namespace internal
} // namespace op

#endif
