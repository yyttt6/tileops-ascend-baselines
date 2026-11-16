/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdint>
#include <string>
#include <map>
#include <algorithm>
#include <cctype>
#include "platform_stub.h"
#include "tiling/platform/platform_ascendc.h"

namespace platform_ascendc {
PlatformAscendC* PlatformAscendCManager::platformInfo = nullptr;
std::mutex PlatformAscendCManager::platformInitMtx;
SocVersion PlatformAscendCManager::SocVersionMap(const char *socVersionStr)
{
    return SocVersion::ASCEND910B;
}
fe::PlatFormInfos* PlatformAscendCManager::PlatformAscendCInit(const char *customSocVersion)
{
    return nullptr;
}
PlatformAscendC* PlatformAscendCManager::PlatformAscendCManagerInit(const char *customSocVersion)
{
    fe::PlatFormInfos pfs;
    static PlatformAscendC pfc(&pfs);
    platformInfo = &pfc;
    return nullptr;
}
SocVersion PlatformAscendC::GetSocVersion(void) const {
    std::string socVersionStr;
    const auto ret = this->platformInfo_->GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr);
    if (!ret) {
        return SocVersion::RESERVED_VERSION;
    }
    static std::map<std::string, SocVersion> convertMap = {
        {"Ascend310P", SocVersion::ASCEND310P},
        {"Ascend910", SocVersion::ASCEND910},
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend910_93", SocVersion::ASCEND910B},
        {"Ascend950", SocVersion::ASCEND950},
        {"Ascend310B", SocVersion::ASCEND310B},
        {"MC62CM12A", SocVersion::MC62CM12A},
    };
    auto it = convertMap.find(socVersionStr);
    if (it != convertMap.end()) {
        return it->second;
    }
    return SocVersion::RESERVED_VERSION;
}

NpuArch PlatformAscendC::GetCurNpuArch(void) const
{
    std::string npuArchStr;
    bool ret = this->platformInfo_->GetPlatformResWithLock("version", "NpuArch", npuArchStr);
    int32_t npuArchInt = std::atoi(npuArchStr.c_str());
    return static_cast<NpuArch>(npuArchInt);
}

void PlatformAscendC::GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const {
    const fe::LocalMemType localType = static_cast<fe::LocalMemType>(memType);
    this->platformInfo_->GetLocalMemSize(localType, size);
    // only ascend910B need UB/L1 local reserved buf for kfc
    if ((memType == CoreMemType::UB || memType == CoreMemType::L1)
         && GetSocVersion() == SocVersion::ASCEND910B) {
        size -= 256;
    }
}

uint32_t PlatformAscendC::GetCoreNumAiv(void) const {
    return this->platformInfo_->GetCoreNum();
}

uint32_t PlatformAscendC::GetVecRegLen(void) const
{
    std::string sizeStr; 
    bool ret = this->platformInfo_->GetPlatformResWithLock("AICoreSpec", "vector_reg_width", sizeStr);
    if(!ret){
        return 0u;
    }
    bool isDecimalNumber = !sizeStr.empty() && std::all_of(sizeStr.begin(), sizeStr.end(), [](unsigned char ch){ return std::isdigit(ch); });
    if(isDecimalNumber){
        uint64_t size = std::stoull(sizeStr);
        if(size > UINT32_MAX){
            return 0u;
        }
        return static_cast<uint32_t>(size);
    }else{
        return 0u;
    }
}
} // namespace platform_ascendc
