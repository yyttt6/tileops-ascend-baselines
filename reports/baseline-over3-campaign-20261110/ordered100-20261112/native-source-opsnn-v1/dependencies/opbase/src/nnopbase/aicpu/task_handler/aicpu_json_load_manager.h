/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_JSON_LOAD_MANAGER_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_JSON_LOAD_MANAGER_H_

#include <nlohmann/json.hpp>
#include <string>
#include <mutex>
#include "aclnn/aclnn_base.h"
#include "acl/acl_rt.h"
#include "ops_struct.h"
#include "opdev/aicpu/aicpu_task.h"

namespace op {
namespace internal {

class JsonLoadManger {
public:
    static aclnnStatus LoadAicpuBinaryFromJson();
    static aclrtBinHandle GetAicpuBinaryHandle() { return aicpuBinHandle_; }
    static aclnnStatus LoadTfBinaryFromJson();
    static aclrtBinHandle GetTfBinaryHandle() { return tfBinHandle_; }
    static bool IsSupportedNewLaunch() { return isSupportNewLaunch_; }
    static aclnnStatus SetSupportedNewLaunchFlag();
    typedef struct {
        bool hasLoad;
        aclrtBinHandle binHandle;
    } CustomBinManager;
    enum class OpPackageType {
        CUSTOM = 0,
        BUILTIN = 1,
    };
    struct OpJsonFileInfo {
        std::string opsRegisterName;
        nlohmann::json opJson;
        OpPackageType packageType;
    };
    struct OpRegisterInfo {
        std::string opsRootPath;
        OpPackageType packageType;
    };

    static aclrtBinHandle GetAicpuCustBinaryHandle(const std::string& kernelSoPath)
    {
        return customBinhandleInfos_[kernelSoPath].binHandle;
    }
    static aclnnStatus LoadAicpuCustBinaryFromJson(const std::string& opType, const std::string& kernelSoName,
                                                   std::string& kernelSoPath);
    static aclnnStatus CustJsonLoadAndParse();
    static bool FindAndGetInCustomRegistry(const std::string& opType, std::string& kernelSo, std::string& functionName);

private:
    JsonLoadManger() = default;
    ~JsonLoadManger();
    JsonLoadManger(const JsonLoadManger&) = delete;
    JsonLoadManger(JsonLoadManger&&) = delete;
    JsonLoadManger& operator=(const JsonLoadManger&) = delete;
    JsonLoadManger& operator=(JsonLoadManger&&) = delete;
    static aclnnStatus LoadBinaryFromJson(const std::string& opsPath, aclrtBinHandle& binHandle,
                                          const bool isCust = false);
    static bool ReadCustJsonFile(const std::string& opsRegisterName, const std::string& customJsonPath,
                                 const OpPackageType packageType);
    static bool ReadCustOpInfoFromJsonFile(const std::string& path);
    static bool GetCustomOppPathFromVendors(std::string& customOppPath);
    static bool ReadBuiltinOpInfoFromJsonFile(const std::string& oppPath);
    static void FillCustOpInfos(std::string opsRegisterName, const OpInfoDescs& infoDesc,
                                const OpPackageType packageType);
    static aclnnStatus ParseCustOpInfo();
    static bool ReadBytesFromBinaryFile(const std::string& fileName, std::vector<char>& buffer);
    static std::shared_ptr<std::vector<char>> GetOrCreateBuffer(const std::string& filePath);
    static bool hasAicpuLoadBin_;
    static aclrtBinHandle aicpuBinHandle_;
    static bool hasTfLoadBin_;
    static aclrtBinHandle tfBinHandle_;
    static std::mutex tfBinLoadMutex_;
    static std::mutex aicpuBinLoadMutex_;
    static bool isSupportNewLaunch_;
    static std::string socVersion_;
    static std::mutex getSocVersionMutex_;
    static bool aicpuCustLoadFlag_;
    static std::mutex custMutex_;
    static std::mutex aicpuCustBinLoadMutex_;
    static std::mutex bufferCacheMutex_;
    // AICPU package repository, including custom and builtin json packages.
    static std::vector<OpJsonFileInfo> custOpJsonInfo_;
    // store operator's name and detailed information
    static std::map<std::string, OpFullInfo> customOpsInfos_;
    // store operator's name and operator's register
    static std::map<std::string, OpRegisterInfo> custRegisterInfos_;
    static std::map<std::string, CustomBinManager> customBinhandleInfos_;
    static std::map<std::string, std::shared_ptr<std::vector<char>>> bufferCache_; // 文件路径到buffer的映射
};

} // namespace internal
} // namespace op
#endif // OP_API_COMMON_INC_OPDEV_AICPU_JSON_LOAD_MANAGER_H_
