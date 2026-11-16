/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_stub.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include <iostream>
#include <map>

uint32_t fe::PlatformInfoManager::InitializePlatformInfo() { return 0U; }

uint32_t fe::PlatformInfoManager::GetPlatformInfos(const std::string SoCVersion, fe::PlatFormInfos& platform_info,
                                                   fe::OptionalInfos& opti_compilation_info)
{
    // platform_info.Init();
    return 0;
}

uint32_t fe::PlatformInfoManager::UpdatePlatformInfos(const string& soc_version, fe::PlatFormInfos& platform_info)
{
    return 0;
}

std::map<std::string, std::vector<std::string>> fe::PlatFormInfos::GetAICoreIntrinsicDtype()
{
    std::map<std::string, std::vector<std::string>> res{
        {"Intrinsic_mmad",
         {"u32u8u8", "s32s8s8", "s32u8s8", "f16f16f16", "f32f16f16", "f16f16u2", "u8", "s8", "f162f16", "f162f32",
          "f16u2", "u8s8"}}};
    return res;
}

thread_local std::shared_ptr<PlatformInfoStub> PlatformInfoStub::instance_;

bool fe::PlatFormInfos::GetPlatformRes(const std::string& label, const std::string& key, std::string& val)
{
    std::string shortSoCVersion;
    std::string socVersion;
    PlatformInfoStub::GetInstance()->GetSoCVersion(shortSoCVersion, socVersion);
    // mock this: GetPlatformRes("version", "Short_SoC_version", socVersionStr);
    if (label == "version" && key == "Short_SoC_version") {
        val = shortSoCVersion;
        return true;
    }
    if (label == "version" && key == "SoC_version") {
        val = socVersion;
        return true;
    }
    // mock this: GetPlatformRes("AICoreSpec", "ubblock_size", blockSizeStr)
    if (label == "AICoreSpec" && key == "ubblock_size") {
        val = "32";
        return true;
    }
    if (label == "SoCInfo" && key == "ffts_mode") {
        val = "ffts-plus";
        return true;
    }
    return false;
}

void fe::PlatFormInfos::SetPlatformResWithLock(const std::string& label, std::map<std::string, std::string>& res)
{
    return;
}

void fe::PlatFormInfos::SetCoreNumByCoreType(const std::string& core_type)
{
    if (core_type == "AiCore") {
        core_num_ = 64;
    } else if (core_type == "VectorCore") {
        core_num_ = 32;
    } else {
        core_num_ = 0;
    }
    return;
}

uint32_t fe::PlatFormInfos::GetCoreNum() const { return core_num_; }

uint32_t fe::PlatformInfoManager::InitRuntimePlatformInfos(const std::string& soCVersion) { return 0; }

fe::PlatformInfoManager& fe::PlatformInfoManager::Instance()
{
    static fe::PlatformInfoManager pf;
    return pf;
}

fe::PlatformInfoManager::PlatformInfoManager() {}
fe::PlatformInfoManager::~PlatformInfoManager() {}

fe::PlatformInfoManager& fe::PlatformInfoManager::GeInstance()
{
    static fe::PlatformInfoManager pf;
    return pf;
}

bool fe::PlatFormInfos::GetPlatformResWithLock(const std::string& label, const std::string& key, std::string& val)
{
    if (label == "DtypeMKN" && key == "Default") {
        val = "16,16,16";
        return true;
    }
    return fe::PlatFormInfos::GetPlatformRes(label, key, val);
}

bool fe::PlatFormInfos::GetPlatformResWithLock(const std::string& label, std::map<std::string, std::string>& res)
{
    if (label == "DtypeMKN") {
        res = {{"DT_UINT8", "16,32,16"}, {"DT_INT8", "16,32,16"},   {"DT_INT4", "16,64,16"},
               {"DT_INT2", "16,128,16"}, {"DT_UINT2", "16,128,16"}, {"DT_UINT1", "16,256,16"}};
    }
    return true;
}

uint32_t fe::PlatformInfoManager::GetPlatformInfoWithOutSocVersion(fe::PlatFormInfos& platform_info,
                                                                   fe::OptionalInfos& opti_compilation_info)
{
    return 0U;
}
