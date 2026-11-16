/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/platform.h"
#include <map>
#include <vector>
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"
#include "platform/platform_info.h"
#include "opdev/fast_vector.h"
#include "acl/acl_rt.h"

using namespace std;

namespace op {
constexpr size_t SOC_SPEC_COUNT = 3;
constexpr size_t SOC_SPEC_ABILITY_DEFAULT_COUNT = 10;
constexpr int64_t DEFAULT_BLOCK_SIZE = 32;
constexpr size_t NPU_ARCH_VAL_MAX_LEN = 10;

using SocSpecAbilityVector = std::vector<SocSpecAbility>;

class PlatformThreadLocalCtx {
public:
    PlatformThreadLocalCtx() {};

    ~PlatformThreadLocalCtx()
    {
        for (auto p : platformInfoList_) {
            delete p;
        }
    };

    const PlatformInfo& GetPlatformInfo();

private:
    std::vector<PlatformInfo*> platformInfoList_;
    PlatformInfo invalidPlatform_;
};

static thread_local PlatformThreadLocalCtx g_tPlatformCtx;

class PlatformInfoImpl {
public:
    explicit PlatformInfoImpl(fe::PlatFormInfos* platformOriginInfo);

    ~PlatformInfoImpl();

    bool CheckAiCoreInstSupport(SocSpec socSpec, SocSpecAbility specAbility) const;

    SocVersion GetSocVersion() const;

    const std::string& GetSocLongVersion() const;

    int64_t GetBlockSize() const;

    uint32_t GetCubeCoreNum() const;

    uint32_t GetVectorCoreNum() const;

    fe::PlatFormInfos* GetPlatformInfos() const;

    bool GetFftsPlusMode() const;

    NpuArch GetCurNpuArch() const;

private:
    void InitAiCoreInstFromOriginPlatformInfo();

    void InitAiCoreInstAbility();

    void InitInstMMadSocSpecAbility(map<string, vector<string>>& aiCoreInstMap);

    void InitAiCoreSpec();

    void InitSocVersion();

    void InitCoreNum();

    void InitSoCInfo();

    void InitNpuArch();

private:
    fe::PlatFormInfos* platformOriInfo_;
    SocVersion socVersion_ = SocVersion::RESERVED_VERSION;
    std::string socLogVersion_;
    int64_t blockSize_ = DEFAULT_BLOCK_SIZE;
    uint32_t cubeCoreNum_{0};
    uint32_t vectorCoreNum_{0};
    std::vector<SocSpecAbilityVector> aiCoreInstAbility_;
    bool fftsPlusMode_{false};
    NpuArch npuArch_{NpuArch::DAV_RESV};
};

PlatformInfoImpl::PlatformInfoImpl(fe::PlatFormInfos* platformOriginInfo) : platformOriInfo_(platformOriginInfo)
{
    InitAiCoreInstAbility();
    InitAiCoreInstFromOriginPlatformInfo();
    InitCoreNum();
    InitNpuArch();
}

PlatformInfoImpl::~PlatformInfoImpl()
{
    if (platformOriInfo_ != nullptr) {
        delete platformOriInfo_;
    }
}

void PlatformInfoImpl::InitAiCoreInstAbility()
{
    for (size_t i = 0; i < SOC_SPEC_COUNT; i++) {
        SocSpecAbilityVector ability;
        aiCoreInstAbility_.push_back(ability);
    }
}

bool PlatformInfoImpl::CheckAiCoreInstSupport(op::SocSpec socSpec, op::SocSpecAbility specAbility) const
{
    CHECK_RET(aiCoreInstAbility_.size() >= static_cast<size_t>(socSpec), false);
    auto socSpecIdx = static_cast<int32_t>(socSpec);
    auto& specAbilityVector = aiCoreInstAbility_[socSpecIdx];
    auto it = std::find(specAbilityVector.begin(), specAbilityVector.end(), specAbility);
    return it != specAbilityVector.end();
}

SocVersion PlatformInfoImpl::GetSocVersion() const { return socVersion_; }

const std::string& PlatformInfoImpl::GetSocLongVersion() const { return socLogVersion_; }

int64_t PlatformInfoImpl::GetBlockSize() const { return blockSize_; }

uint32_t PlatformInfoImpl::GetCubeCoreNum() const { return cubeCoreNum_; }

uint32_t PlatformInfoImpl::GetVectorCoreNum() const { return vectorCoreNum_; }

bool PlatformInfoImpl::GetFftsPlusMode() const { return fftsPlusMode_; }

NpuArch PlatformInfoImpl::GetCurNpuArch() const { return npuArch_; }

fe::PlatFormInfos* PlatformInfoImpl::GetPlatformInfos() const { return platformOriInfo_; }

void PlatformInfoImpl::InitAiCoreInstFromOriginPlatformInfo()
{
    OP_LOGI("Entering InitFromOriginPlatformInfo");
    map<string, vector<string>> aiCoreInstMap = platformOriInfo_->GetAICoreIntrinsicDtype();
    OP_LOGI("Call GetAICoreIntrinsicDtype success.");
    // aiCoreInstAbility_'s value index should match SocSpec enum.
    InitInstMMadSocSpecAbility(aiCoreInstMap);
    InitSocVersion();
    InitAiCoreSpec();
    InitSoCInfo();
}

void PlatformInfoImpl::InitInstMMadSocSpecAbility(map<string, vector<string>>& aiCoreInstMap)
{
    OP_LOGI("Start InitInstMMadSocSpecAbility.");
    // all ability:
    // u32u8u8,s32s8s8,s32u8s8,f16f16f16,f32f16f16,f16f16u2,u8,s8,f162f16,f162f32,f16u2,u8s8,h322f32,f322f32
    map<string, SocSpecAbility> convertMap = {{"u32u8u8", SocSpecAbility::INST_MMAD_U32U8U8},
                                              {"s32s8s8", SocSpecAbility::INST_MMAD_S32S8S8},
                                              {"s32u8s8", SocSpecAbility::INST_MMAD_S32U8S8},
                                              {"f16f16f16", SocSpecAbility::INST_MMAD_F16F16F16},
                                              {"f32f16f16", SocSpecAbility::INST_MMAD_F32F16F16},
                                              {"f16f16u2", SocSpecAbility::INST_MMAD_F16F16U2},
                                              {"u8", SocSpecAbility::INST_MMAD_U8},
                                              {"s8", SocSpecAbility::INST_MMAD_S8},
                                              {"f162f16", SocSpecAbility::INST_MMAD_F162F16},
                                              {"f162f32", SocSpecAbility::INST_MMAD_F162F32},
                                              {"f16u2", SocSpecAbility::INST_MMAD_F16U2},
                                              {"u8s8", SocSpecAbility::INST_MMAD_U8S8},
                                              {"h322f32", SocSpecAbility::INST_MMAD_H322F32},
                                              {"f322f32", SocSpecAbility::INST_MMAD_F322F32}};

    auto instMapIt = aiCoreInstMap.find("Intrinsic_mmad");
    if (instMapIt != aiCoreInstMap.end()) {
        for (auto& instStr : instMapIt->second) {
            auto abilityIt = convertMap.find(instStr);
            if (abilityIt != convertMap.end()) {
                aiCoreInstAbility_[static_cast<::int32_t>(SocSpec::INST_MMAD)].push_back(abilityIt->second);
            } else {
                OP_LOGW("InitInstMMadSocSpecAbility convert failed, inst str: %s", instStr.c_str());
            }
        }
    } else {
        OP_LOGI("Not find 'Intrinsic_mmad'.");
    }
}

void PlatformInfoImpl::InitSocVersion()
{
    string socVersionStr;
    auto ret = platformOriInfo_->GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr);
    if (!ret) {
        OP_LOGW("InitFromOriginPlatformInfo failed, get soc version failed.");
        socVersion_ = SocVersion::RESERVED_VERSION;
    } else {
        map<string, SocVersion> convertMap = {
            {"Ascend910", SocVersion::ASCEND910},       {"Ascend910B", SocVersion::ASCEND910B},
            {"Ascend910_93", SocVersion::ASCEND910_93}, {"Ascend910_95", SocVersion::ASCEND910_95},
            {"Ascend950", SocVersion::ASCEND950},       {"Ascend910E", SocVersion::ASCEND910E},
            {"Ascend310", SocVersion::ASCEND310},       {"Ascend310B", SocVersion::ASCEND310B},
            {"Ascend310C", SocVersion::ASCEND310C},     {"Ascend310P", SocVersion::ASCEND310P},
            {"Ascend350", SocVersion::ASCEND350},       {"Ascend610Lite", SocVersion::ASCEND610LITE},
            {"KirinX90", SocVersion::KIRINX90},         {"Kirin9030", SocVersion::KIRIN9030}};
        auto it = convertMap.find(socVersionStr);
        if (it != convertMap.end()) {
            socVersion_ = it->second;
        } else {
            OP_LOGW("ConvertSocVersion convert failed, socVersionStr: %s", socVersionStr.c_str());
            socVersion_ = SocVersion::RESERVED_VERSION;
        }
    }

    socLogVersion_ = "unknownVersion";
    ret = platformOriInfo_->GetPlatformResWithLock("version", "SoC_version", socLogVersion_);
    if (!ret) {
        OP_LOGW("InitFromOriginPlatformInfo failed, get soc long version failed.");
    }
    OP_LOGI("soc long version is %s, soc short version is %s", socLogVersion_.c_str(), socVersionStr.c_str());
}

void PlatformInfoImpl::InitAiCoreSpec()
{
    string blockSizeStr;
    auto ret = platformOriInfo_->GetPlatformResWithLock("AICoreSpec", "ubblock_size", blockSizeStr);
    OP_LOGI("GetPlatformResWithLock get ubblock_size ret is: %d, ubblock_size is: %ld", ret, blockSize_);
    CHECK_RET(ret, );
    try {
        int num = stoi(blockSizeStr);
        blockSize_ = num;
        OP_LOGI("InitAiCoreSpec success, block size is: %ld", blockSize_);
    } catch (const invalid_argument& e) {
        OP_LOGW("InitAiCoreSpec Failed, ubblock_size is not a number string: %s, use default 32.",
                blockSizeStr.c_str());
    } catch (const out_of_range& e) {
        OP_LOGW("InitAiCoreSpec Failed, ubblock_size is out of int range: %s, use default 32.", blockSizeStr.c_str());
    }
}

// init coreNum to the coreType
void PlatformInfoImpl::InitCoreNum()
{
    fe::PlatFormInfos plat = *platformOriInfo_;
    plat.SetCoreNumByCoreType("AiCore");
    cubeCoreNum_ = plat.GetCoreNum();
    plat.SetCoreNumByCoreType("VectorCore");
    vectorCoreNum_ = plat.GetCoreNum();
}

void PlatformInfoImpl::InitSoCInfo()
{
    string fftsModeStr = "ffts-default-not";
    bool res = platformOriInfo_->GetPlatformResWithLock("SoCInfo", "ffts_mode", fftsModeStr);
    fftsPlusMode_ = ((res && fftsModeStr == "ffts-plus") ? true : false);
    OP_LOGI("GetPlatformResWithLock res is %d, get ffts_mode is: %s", static_cast<int>(res), fftsModeStr.c_str());
}

void PlatformInfoImpl::InitNpuArch()
{
    char archVal[NPU_ARCH_VAL_MAX_LEN] = {0};
    OP_CHECK(rtGetSocSpec("version", "NpuArch", archVal, NPU_ARCH_VAL_MAX_LEN) == RT_ERROR_NONE,
             OP_LOGW("call rtGetSocSpec failed"), return);
    std::string archStrVal(archVal);
    uint32_t npuArchVal = 0;
    try {
        npuArchVal = static_cast<uint32_t>(std::stoul(archStrVal));
    } catch (const std::exception& e) {
        OP_LOGW("stoul %s to uint32_t value failed, reason: %s", archStrVal.c_str(), e.what());
        npuArch_ = NpuArch::DAV_RESV;
        return;
    }
    OP_LOGI("Npu arch get from rtGetSocSpec: %s, change to uint32_t: %u", archVal, npuArchVal);
    npuArch_ = static_cast<NpuArch>(npuArchVal);
}

bool PlatformInfo::CheckSupport(op::SocSpec socSpec, op::SocSpecAbility ability) const
{
    if (impl_ != nullptr) {
        return impl_->CheckAiCoreInstSupport(socSpec, ability);
    }
    return false;
}

PlatformInfo::~PlatformInfo()
{
    if (impl_ != nullptr) {
        delete impl_;
    }
}

bool PlatformInfo::Valid() const { return valid_; }

void PlatformInfo::SetPlatformImpl(op::PlatformInfoImpl* impl)
{
    impl_ = impl;
    valid_ = true;
}

SocVersion PlatformInfo::GetSocVersion() const
{
    CHECK_RET(impl_ != nullptr, SocVersion::RESERVED_VERSION);
    return impl_->GetSocVersion();
}

const std::string& PlatformInfo::GetSocLongVersion() const
{
    static std::string unknownVersion = "unknownVersion";
    CHECK_RET(impl_ != nullptr, unknownVersion);
    return impl_->GetSocLongVersion();
}

int32_t PlatformInfo::GetDeviceId() const { return deviceId_; }

int64_t PlatformInfo::GetBlockSize() const
{
    CHECK_RET(impl_ != nullptr, DEFAULT_BLOCK_SIZE);
    return impl_->GetBlockSize();
}

uint32_t PlatformInfo::GetCubeCoreNum() const
{
    CHECK_RET(impl_ != nullptr, 0);
    return impl_->GetCubeCoreNum();
}

uint32_t PlatformInfo::GetVectorCoreNum() const
{
    CHECK_RET(impl_ != nullptr, 0);
    return impl_->GetVectorCoreNum();
}

bool PlatformInfo::GetFftsPlusMode() const
{
    CHECK_RET(impl_ != nullptr, false);
    return impl_->GetFftsPlusMode();
}

NpuArch PlatformInfo::GetCurNpuArch() const
{
    CHECK_RET(impl_ != nullptr, NpuArch::DAV_RESV);
    return impl_->GetCurNpuArch();
}

fe::PlatFormInfos* PlatformInfo::GetPlatformInfos() const
{
    CHECK_RET(impl_ != nullptr, nullptr);
    return impl_->GetPlatformInfos();
}

const PlatformInfo& PlatformThreadLocalCtx::GetPlatformInfo()
{
    int32_t deviceId;
    auto ret = aclrtGetDevice(&deviceId);
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Get device id failed when do GetCurrentPlatformInfo."),
             return invalidPlatform_);
    while (platformInfoList_.size() <= static_cast<size_t>(deviceId)) {
        PlatformInfo* info = new PlatformInfo(platformInfoList_.size());
        platformInfoList_.push_back(info);
    }
    OP_LOGI("GetCurrentPlatformInfo, get current device is: %d, tPlatformInfoList.size(): %lu", deviceId,
            platformInfoList_.size());
    if (!platformInfoList_[deviceId]->Valid()) {
        fe::PlatFormInfos* platformOriginInfo = new fe::PlatFormInfos();
        auto getPlatformRet = fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(
            deviceId, *platformOriginInfo);
        if (getPlatformRet != 0) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "Call PlatformInfoManager::GeInstance().GetPlatformInstanceByDevice failed.");
            delete platformOriginInfo;
        } else {
            OP_LOGI("GetRuntimePlatformInfosByDevice success.");
            PlatformInfoImpl* platformInfoImpl = new PlatformInfoImpl(platformOriginInfo);
            platformInfoList_[deviceId]->SetPlatformImpl(platformInfoImpl);
        }
    }
    return *platformInfoList_[deviceId];
}

const PlatformInfo& GetCurrentPlatformInfo() { return g_tPlatformCtx.GetPlatformInfo(); }

ge::AscendString ToString(SocVersion socVersion)
{
    static const std::map<SocVersion, std::string> kSocVersionMap = {
        {SocVersion::ASCEND910, "Ascend910"},               // SocVersion::ASCEND910
        {SocVersion::ASCEND910B, "Ascend910B"},             // SocVersion::ASCEND910B
        {SocVersion::ASCEND910_93, "Ascend910_93"},         // SocVersion::ASCEND910_93
        {SocVersion::ASCEND950, "Ascend950"},               // SocVersion::ASCEND950
        {SocVersion::ASCEND910E, "Ascend910E"},             // SocVersion::ASCEND910E
        {SocVersion::ASCEND310, "Ascend310"},               // SocVersion::ASCEND310
        {SocVersion::ASCEND310B, "Ascend310B"},             // SocVersion::ASCEND310B
        {SocVersion::ASCEND310C, "Ascend310C"},             // SocVersion::ASCEND310C
        {SocVersion::ASCEND350, "Ascend350"},               // SocVersion::ASCEND350
        {SocVersion::ASCEND310P, "Ascend310P"},             // SocVersion::ASCEND310P
        {SocVersion::ASCEND610LITE, "Ascend610Lite"},       // SocVersion::ASCEND610LITE
        {SocVersion::KIRINX90, "KirinX90"},                 // SocVersion::KIRINX90
        {SocVersion::KIRIN9030, "Kirin9030"},               // SocVersion::KIRIN9030
        {SocVersion::RESERVED_VERSION, "UnknownSocVersion"} // SocVersion::RESERVED_VERSION
    };
    static const std::string reserved("UnknownSocVersion");
    const auto it = kSocVersionMap.find(socVersion);
    if (it != kSocVersionMap.end()) {
        return ge::AscendString((it->second).c_str());
    } else {
        OP_LOGW("unknown soc version:%d.", static_cast<int32_t>(socVersion));
        return ge::AscendString((reserved).c_str());
    }
}

} // namespace op
