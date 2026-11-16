/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_SOC_H_
#define INDV_SOC_H_

#include <string>
#include <map>
#include <vector>
#include "platform.h"
#include "individual_op_api.h"

namespace nnopbase {
constexpr const char* OPS_SUBPATH_ASCEND910 = "ascend910";
constexpr const char* OPS_SUBPATH_ASCEND910B = "ascend910b";
constexpr const char* OPS_SUBPATH_ASCEND910_93 = "ascend910_93";
constexpr const char* OPS_SUBPATH_ASCEND950 = "ascend950";
constexpr const char* OPS_SUBPATH_ASCEND310P = "ascend310p";
constexpr const char* OPS_SUBPATH_ASCEND310B = "ascend310b";
constexpr const char* OPS_SUBPATH_ASCEND610LITE = "ascend610lite";
constexpr const char* OPS_SUBPATH_ASCEND910_96 = "ascend910_96";
constexpr const char* OPS_SUBPATH_ASCEND350 = "ascend350";

constexpr uint32_t SOC_VERSION_ASCEND910A = 1U;
constexpr uint32_t SOC_VERSION_ASCEND910B = 2U;
constexpr uint32_t SOC_VERSION_ASCEND910_93 = 3U;
constexpr uint32_t SOC_VERSION_ASCEND950 = 4U;
constexpr uint32_t SOC_VERSION_ASCEND310P = 5U;
constexpr uint32_t SOC_VERSION_ASCEND310B = 6U;
constexpr uint32_t SOC_VERSION_ASCEND610Lite = 8U;
constexpr uint32_t SOC_VERSION_ASCEND910_96 = 13U;
constexpr uint32_t SOC_VERSION_ASCEND350 = 16U;
constexpr uint32_t SOC_VERSION_INVALID = 99U;
class IndvSoc {
public:
    static IndvSoc& GetInstance(void);

    const std::map<std::string, uint32_t>& GetSocTypeMap(void) const;

    const std::string& GetCurSocVersion(void);
    bool SupportCurrentSoc(void) const;
    bool UseCoreTypeMagic(void) const;
    bool SupportMc2FusionLaunch(void);
    bool NeedAlignInitValues(void) const;
    bool NeedsExtraMemoryForOverflowDump(void) const;
    bool SupportL0ExceptionDump(void) const;
    bool IsCouplingArch(void) const;
    bool NnopbaseEnableCcuLaunch(const NnopbaseHcclServerType sType);
    bool NnopbaseUseA5Mc2Client(const NnopbaseHcclServerType sType);
    uint32_t* GetNonFiniteCheckSocSupportList(uint32_t& socSupportListLen) const;
    uint32_t GetSocEnum();
    bool IsSupportedSocName(const std::string& name) const;
    uint32_t GetSocEnumByName(const std::string& name) const;
    std::string GetSupportedSocNamesStr() const;
    void Reset(void);
    bool isInit = false;
    std::string socVersion;

private:
    IndvSoc(void);
    void Init(void);

    const std::map<std::string, uint32_t> supportSocMap = {
        {OPS_SUBPATH_ASCEND910, SOC_VERSION_ASCEND910A},        {OPS_SUBPATH_ASCEND910B, SOC_VERSION_ASCEND910B},
        {OPS_SUBPATH_ASCEND910_93, SOC_VERSION_ASCEND910_93},   {OPS_SUBPATH_ASCEND310P, SOC_VERSION_ASCEND310P},
        {OPS_SUBPATH_ASCEND310B, SOC_VERSION_ASCEND310B},       {OPS_SUBPATH_ASCEND950, SOC_VERSION_ASCEND950},
        {OPS_SUBPATH_ASCEND610LITE, SOC_VERSION_ASCEND610Lite}, {OPS_SUBPATH_ASCEND910_96, SOC_VERSION_ASCEND910_96},
        {OPS_SUBPATH_ASCEND350, SOC_VERSION_ASCEND350},
    };
};
} // namespace nnopbase
#endif
