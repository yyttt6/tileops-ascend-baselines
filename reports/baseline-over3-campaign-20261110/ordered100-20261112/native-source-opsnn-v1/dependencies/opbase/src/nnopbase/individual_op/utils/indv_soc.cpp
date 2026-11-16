/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_soc.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <map>
#include <vector>
#include "platform.h"
#include "indv_base.h"
#include "utils/indv_debug_assert.h"

namespace nnopbase {
IndvSoc& IndvSoc::GetInstance(void)
{
    static IndvSoc instance;
    return instance;
}

IndvSoc::IndvSoc(void) {}

void IndvSoc::Init(void)
{
    if (isInit)
        return;
    ge::AscendString curSocVersion = op::ToString(op::GetCurrentPlatformInfo().GetSocVersion());
    socVersion = curSocVersion.GetString();
    std::transform(socVersion.begin(), socVersion.end(), socVersion.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    isInit = true;
}

const std::map<std::string, uint32_t>& IndvSoc::GetSocTypeMap(void) const { return supportSocMap; }

const std::string& IndvSoc::GetCurSocVersion(void)
{
    if (!isInit) {
        Init();
    }
    return socVersion;
}

bool IndvSoc::SupportCurrentSoc(void) const { return supportSocMap.find(socVersion) != supportSocMap.cend(); }

bool IndvSoc::UseCoreTypeMagic(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910B) || (socVersion == OPS_SUBPATH_ASCEND910_93);
}

bool IndvSoc::SupportMc2FusionLaunch(void)
{
    const std::string& curSocVersion = GetCurSocVersion();
    return (curSocVersion == OPS_SUBPATH_ASCEND950) || (curSocVersion == OPS_SUBPATH_ASCEND910_96) ||
           (curSocVersion == OPS_SUBPATH_ASCEND350);
}

bool IndvSoc::NeedAlignInitValues(void) const
{
    // 910B以及后继芯片场景 需要在Memset的输入tensor上做shape对齐
    return (socVersion != OPS_SUBPATH_ASCEND310B) && (socVersion != OPS_SUBPATH_ASCEND310P) &&
           (socVersion != OPS_SUBPATH_ASCEND610LITE) && (socVersion != OPS_SUBPATH_ASCEND910);
}

bool IndvSoc::NeedsExtraMemoryForOverflowDump(void) const { return socVersion == OPS_SUBPATH_ASCEND910B; }

bool IndvSoc::SupportL0ExceptionDump(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910 || socVersion == OPS_SUBPATH_ASCEND310B);
}

bool IndvSoc::IsCouplingArch(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910 || socVersion == OPS_SUBPATH_ASCEND310P);
}

bool IndvSoc::NnopbaseEnableCcuLaunch(const NnopbaseHcclServerType sType)
{
    const bool isEnableCcuLaunch = SupportMc2FusionLaunch() && (sType == NNOPBASE_HCCL_SERVER_TYPE_CCU);
    OP_LOGD("NnopbaseEnableCcuLaunch check, socVersion=%s, sType=%d, isEnableCcuLaunch=%d", socVersion.c_str(),
            static_cast<int>(sType), isEnableCcuLaunch);
    return isEnableCcuLaunch;
}

bool IndvSoc::NnopbaseUseA5Mc2Client(const NnopbaseHcclServerType sType)
{
    const std::string& curSocVersion = GetCurSocVersion();
    const bool isSupportFusionLaunch = SupportMc2FusionLaunch();
    const bool useA5Mc2Client = (isSupportFusionLaunch && ((sType == NNOPBASE_HCCL_SERVER_TYPE_END) ||
                                                           (sType == NNOPBASE_HCCL_SERVER_TYPE_AICPU) ||
                                                           (sType == NNOPBASE_HCCL_SERVER_TYPE_CCU)));
    OP_LOGD("NnopbaseUseA5Mc2Client check, socVersion=%s, sType=%d, isSupportFusionLaunch=%d, useA5Mc2Client=%d",
            curSocVersion.c_str(), static_cast<int>(sType), isSupportFusionLaunch, useA5Mc2Client);
    return useA5Mc2Client;
}

uint32_t* IndvSoc::GetNonFiniteCheckSocSupportList(uint32_t& socSupportListLen) const
{
    static uint32_t socSupportList[] = {SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910_93};
    socSupportListLen = sizeof(socSupportList) / sizeof(uint32_t);
    return socSupportList;
}

uint32_t IndvSoc::GetSocEnum()
{
    const auto curSocVersion = this->GetCurSocVersion();
    const auto socMap = GetSocTypeMap();
    auto iter = socMap.find(curSocVersion);
    if (iter != socMap.cend()) {
        return iter->second;
    }
    return SOC_VERSION_INVALID;
}

bool IndvSoc::IsSupportedSocName(const std::string& name) const
{
    return supportSocMap.find(name) != supportSocMap.cend();
}

uint32_t IndvSoc::GetSocEnumByName(const std::string& name) const
{
    auto iter = supportSocMap.find(name);
    if (iter != supportSocMap.cend()) {
        return iter->second;
    }
    return SOC_VERSION_INVALID;
}

std::string IndvSoc::GetSupportedSocNamesStr() const
{
    std::string result;
    for (const auto& entry : supportSocMap) {
        if (!result.empty()) {
            result += ", ";
        }
        result += entry.first;
    }
    return result;
}

void IndvSoc::Reset(void)
{
    isInit = false;
    socVersion = "";
}
} // namespace nnopbase
