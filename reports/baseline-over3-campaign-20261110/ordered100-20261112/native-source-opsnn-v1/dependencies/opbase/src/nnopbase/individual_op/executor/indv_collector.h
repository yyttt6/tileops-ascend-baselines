/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_COLLECTOR_H_
#define INDV_COLLECTOR_H_

#include <string>
#include "nlohmann/json.hpp"
#include "utils/indv_base.h"
#include "utils/indv_dlist.h"
#include "indv_bininfo.h"
#include "platform/platform_info.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* const SCENE = "scene.info";
const size_t SCENE_VALUE_COUNT = 2U;
const size_t SCENE_KEY_INDEX = 0U;
const size_t SCENE_VALUE_INDEX = 1U;
const char* const SCENE_OS = "os";
const char* const SCENE_ARCH = "arch";

constexpr const char* NNOPBASE_SIMPLIFIED_KEY_MODE_JSON_KEY = "simplifiedKeyMode";
constexpr int32_t NNOPBASE_SIMPLIFIED_KEY_MODE_CUSTOMIZED = 2;

typedef struct {
    DList head;
} RegInfoBucket;

typedef struct {
    RegInfoBucket buckets[NNOPBASE_NORM_MAX_BIN_BUCKETS];
} RegInfoTbl;

typedef struct {
    RegInfoTbl regInfoTbl;
    bool useCoreTypeMagic = false;
    bool isMc2FusionLaunch = false; // 对于950后的芯片，mc2算子使用fusion launch
    std::string oppPath;
    struct timespec collectorTp[NnopbaseCollectorTimeIdx::kEnd];
} NnopbaseBinCollector;

extern NnopbaseBinCollector* gBinCollector;
aclnnStatus NnopbaseCollectorWork(NnopbaseBinCollector* const collector);
NnopbaseBinInfo* NnopbaseCollectorFindBinInfo(NnopbaseRegInfo* const regInfo, const size_t hashKey,
                                              const NnopbaseUChar* const verbose, const uint32_t verbLen,
                                              const StaticKernelPlatformInfo* const platformInfo = nullptr);
void NnopbaseCollectorInsertBinInfo(NnopbaseRegInfo* const regInfo, NnopbaseBinInfo* binInfo);
aclnnStatus NnopbaseCollectorAddBinInfo(const string& key, NnopbaseRegInfo* const regInfo,
                                        const NnopbaseJsonInfo& jsonInfo, const NnopbaseUChar* const verbose,
                                        const uint32_t len);
aclnnStatus NnopbaseCollectorAddRegInfoToTbl(NnopbaseBinCollector* const collector, const NnopbaseJsonInfo& jsonInfo,
                                             const uint64_t hashKey, NnopbaseRegInfo*& reg,
                                             gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollectorAddRepoInfo(NnopbaseBinCollector* const collector, const NnopbaseJsonInfo& jsonInfo,
                                         const std::string& key, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollectorConvertCustomizedVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                      uint32_t* const size);
aclnnStatus NnopbaseCollectorConvertDynamicVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                   uint32_t* const size);
aclnnStatus NnopbaseCollectorConvertStaticVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                  uint32_t* const size);
aclnnStatus NnopbaseSetCollectorSocVersion(NnopbaseBinCollector* collector);

void NnopbaseCollectorOpRegInfoDestroy(NnopbaseRegInfo** regInfo);
aclnnStatus NnopbaseCollectorGcRegInfo(void* data);
aclnnStatus NnopbaseCollectorReadDynamicKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                           const nlohmann::json& binaryInfoConfig,
                                                           const std::string& basePath,
                                                           gert::OppImplVersionTag oppImplVersion,
                                                           const std::string pkgName);

void NnopbaseGetCustomOpApiPath(std::vector<std::string>& basePath);
void NnopbaseGetOppApiPath(std::vector<std::string>& basePath);
void NnopbaseGetCustomOppPath(std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath);
void NnopbaseGetOppPath(NnopbaseBinCollector* const collector,
                        std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath,
                        int32_t& builtInStartIndex);
aclnnStatus NnopbaseGetCurEnvPackageOsAndCpuType(std::string& hostEnvOs, std::string& hostEnvCpu);
aclnnStatus NnopbaseLoadTilingSo(std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath);
aclnnStatus NnopbaseCollectorSetTiling(const NnopbaseJsonInfo& jsonInfo, TilingFun* const tiling,
                                       GenSimplifiedKeyFun* const genSimplifiedKey,
                                       gert::OppImplVersionTag oppImplVersion);
bool NnopbaseReadConfigFile(const std::string& configPath, std::vector<std::string>& subPath);
NnopbaseUChar* NnopbaseCollectorGenStaticKey(NnopbaseUChar* verKey, const NnopbaseRegInfoKey* const regInfoKey,
                                             const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                             const aclTensor* tensors[], const NnopbaseAttrAddr* attrs[],
                                             const int64_t implMode, const int64_t deterMin,
                                             const int64_t* const vDepend, const bool usingStride);
const char* NnopbaseCollectorGetStaticKernelBin(const NnopbaseChar* const opType, const uint64_t key,
                                                const NnopbaseUChar* verbose, const uint32_t verbLen,
                                                const StaticKernelPlatformInfo* const platformInfo = nullptr);
aclnnStatus NnopbaseCollectorGetStaticKernelPathAndReadConfig(NnopbaseBinCollector* const collector,
                                                              const std::string& basePath = "");
aclnnStatus NnopbaseCollectorDeleteStaticBins(NnopbaseRegInfo* regInfo);
void NnopbaseSplitStr(const std::string& configPath, const std::string& pattern, std::vector<std::string>& subPaths);
aclnnStatus NnopbaseCollectorReadDebugKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                         nlohmann::json& binaryInfoConfig, const std::string& basePath,
                                                         gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollectorGetDynamicKernelPathAndReadConfig(
    NnopbaseBinCollector* const collector, const std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath,
    int32_t builtInStartIndex);
aclnnStatus NnopbaseCollectorReadStaticKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                          nlohmann::json& binaryInfoConfig, const std::string& basePath,
                                                          gert::OppImplVersionTag oppImplVersion);
NnopbaseUChar* NnopbaseBeyond8ByteCopy(const int32_t start, const int32_t end, const NnopbaseChar* const strKey,
                                       NnopbaseUChar* verKey);
aclnnStatus NnopbaseCollectorOpRegInfoInit(NnopbaseRegInfo* regInfo, const NnopbaseJsonInfo& jsonInfo,
                                           const uint64_t hashKey, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseUpdateStaticJsonInfo(nlohmann::json& binInfo, NnopbaseJsonInfo& jsonInfo);

aclnnStatus NnopbaseUpdateStaticBinJsonInfos(NnopbaseBinCollector* const collector, const NnopbaseChar* const opType);
aclnnStatus NnopbaseRefreshStaticKernelInfos(NnopbaseBinCollector* const collector);
aclnnStatus UpdateStaticJsonExtraInfo(NnopbaseJsonInfo& jsonInfo);
void SetExtraKernelInfoToBin(const NnopbaseJsonInfo& jsonInfo, std::unique_ptr<NnopbaseBinInfo>& binInfo);

static inline aclnnStatus NnopbaseCollectorInit(NnopbaseBinCollector* collector)
{
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        DoubleListInit(&collector->regInfoTbl.buckets[i].head);
    }
    return NnopbaseSetCollectorSocVersion(collector);
}

static inline void NnopbaseCollectorInitBinTbl(BinTbl* binTbl)
{
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        DoubleListInit(&binTbl->buckets[i].head);
        binTbl->buckets[i].isVisit = false;
    }
}

NnopbaseRegInfo* NnopbaseCollectorFindRegInfoInTbl(const NnopbaseBinCollector* const collector,
                                                   const NnopbaseChar* const opType, const uint64_t hashKey);
#ifdef __cplusplus
}
#endif
#endif
