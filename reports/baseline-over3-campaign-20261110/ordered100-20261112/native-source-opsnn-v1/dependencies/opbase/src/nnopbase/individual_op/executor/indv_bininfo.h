/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_BIN_INFO_H_
#define INDV_BIN_INFO_H_

#include <string>
#include "securec.h"
#include "utils/indv_dlist.h"
#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_hash.h"
#include "utils/indv_types.h"
#include "indv_args.h"
#include "opdev/op_errno.h"
#include "nlohmann/json.hpp"
#include "register/op_impl_registry.h"
#include "kernel_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NNOPBASE_FILE_PATH_MAX_LEN = 4096;
static const std::string NNOPBASE_BUILD_TYPE = "debug";

static const std::string MEMSET_DESC_JSON_COMMON_DIR = "/built-in/op_impl/ai_core/tbe/kernel/config/";
static const std::string MEMSET_BINARY_JSON_COMMON_DIR = "/built-in/op_impl/ai_core/tbe/kernel/";
static const std::string INDV_MEMSET_TBE_FILE_NAME = "/mem_set.json";
static const std::string INDV_MEMSET_ASC_FILE_NAME = "/mem_set_apt.json";
static const std::string NNOPBASE_MEMSET_OP_NAME = "MemSet";

typedef struct {
    DList head;
    bool isVisit;
} BinInfoBucket;

typedef struct {
    BinInfoBucket buckets[NNOPBASE_NORM_MAX_BIN_BUCKETS];
} BinTbl;

typedef struct {
    uint64_t hashKey;
    std::string opType;
} NnopbaseRegInfoKey;

typedef struct {
    NnopbaseRegInfoKey key;
    TilingFun tiling;
    GenSimplifiedKeyFun genSimplifiedKey = nullptr;
    DoubleListNode dllNode;
    BinTbl binTbl;
    bool isActive;
    bool hasStaticShapeBin = false;
    bool customizedSimplifiedKey = false;
} NnopbaseRegInfo;

struct BinInfoKey {
    size_t hashKey = 0U;
    std::vector<NnopbaseUChar> verbose;
    uint32_t len = 0U;
    uint32_t bufLen = 0U;
};

enum NnopbaseTaskRation {
    kRation10 = 0, // AIC:AIV = 1:0
    kRation01,     // AIC:AIV = 0:1
    kRation11,     // AIC:AIV = 1:1
    kRation12,     // AIC:AIV = 1:2
    kRationEnd
};

enum NnopbaseLoadBinInfoType { kDynamicInfo = 0, kStaticBinInfo, kTypeEnd };

struct MixRationParam {
    CoreType coreType = kMix;
    NnopbaseTaskRation taskRation = kRationEnd;
    bool crossCoreSync = true;
};

struct NnopbaseDfxInfo {
    bool isPrintEnable = false;
    bool isAssertEnable = false;
    bool isTimeStampEnable = false;
};

struct NnopbaseCoreNum {
    uint32_t aicNum = 0U;
    uint32_t aivNum = 0U;
};

struct StaticKernelPlatformInfo {
    NnopbaseCoreNum coreNum;
    int8_t deterministicLevel = -1;
};

struct StaticKernelJsonRunInfo {
    uint32_t aicpuNumBlocks = 0U;
    uint32_t numBlocks = 0U;
    uint32_t dynUBufSize = 0U;
    uint32_t scheduleMode = 0U;
    uint32_t tilingCond = 0U;
    uint64_t tilingKey = 0U;
    bool clearAtomic = false;
    size_t workspaceSizesNum = 0U;
    std::vector<size_t> workspaceSizes;
    std::vector<NnopbaseUChar> tilingData;
};

struct ExtraKernelDesc {
    std::unique_ptr<StaticKernelPlatformInfo> platformInfo = nullptr; // 编译静态kernel时的平台描述
    std::unique_ptr<StaticKernelJsonRunInfo> runInfo = nullptr;       // 静态kernel的运行数据描述
};

typedef struct {
    std::vector<std::string> keys; // "simplifiedKey"
    std::string kernelName;
    std::string opType;
    NnopbaseChar path[NNOPBASE_FILE_PATH_MAX_LEN] = {}; // real path of "binPath"
    size_t workspaceSizes[NNOPBASE_NORM_MAX_WORKSPACE_NUMS] = {};
    size_t workspaceSizeNum = 0U;
    CoreType coreType = kCoreTypeEnd;
    bool isStaticShape = false;           // true: include "staticList"
    bool customizedSimplifiedKey = false; // true: "simplifiedKeyMode" is 2
    NnopbaseLoadBinInfoType loadBinInfoType = kDynamicInfo;
    uint32_t numBlocks = 0U;
    uint32_t multiKernelType = 0U;
    std::unordered_map<uint64_t, MixRationParam> tilingKeyInfo;
    const NnopbaseUChar* bin = nullptr;
    uint32_t binLen = 0U;
    NnopbaseDfxInfo dfxInfo;
    size_t debugBufSize = 0U;
    NnopbaseTaskRation taskRation = kRationEnd;
    ExtraKernelDesc extraKernelDesc;
} NnopbaseJsonInfo;

struct NnopbaseInitValueInfo {
    size_t irIndex;
    op::DataType dtype;
    float_t floatValue;
    int64_t intValue;
    size_t tensorDataSize;
};

struct MemsetOpBinInfo {
    ~MemsetOpBinInfo()
    {
        if (bin != nullptr) {
            delete[] bin;
            bin = nullptr;
        }
    }
    const NnopbaseChar* opType = nullptr;
    const NnopbaseUChar* bin = nullptr;
    std::string binPath;
    void* binHandle = nullptr;
    uint32_t binLen = 0U;
    uint32_t magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    std::string binFileName;
    std::string kernelName;
    CoreType coreType = kCoreTypeEnd;
    uint32_t opParaSize = 0U;
    bool loadFuncHandleByTilingKey = false;
};

struct MemsetOpInfo {
    NnopbaseRTArgsExt argsExt;
    std::string compileInfo;
    std::string memSetJsonPath;
    TilingFun tilingFunc = nullptr;
    std::shared_ptr<MemsetOpBinInfo> binInfo = nullptr;
    NnopbaseKernelRunContextExt contextExt;
    NnopbaseAsyncAnyValue* tilingParseContextValue = nullptr;
    NnopbaseKernelRunContext* tilingParseContext = nullptr;
    uint8_t tilingData[NNOPBASE_TILING_DATA_STRUCT_SIZE] = {};
    uint8_t workspacesSizes[NNOPBASE_WORKSPACE_STRUCT_SIZE] = {};
    uint64_t* tilingKey = nullptr;
    uint32_t* numBlocks = nullptr;
    uint32_t* scheMode = nullptr;
    uint32_t* dynUBufSize = nullptr;
};

typedef struct {
    BinInfoKey binInfoKey;
    void* regInfo = nullptr;
    const NnopbaseUChar* bin;
    std::string simplifiedKey;
    std::string binPath;
    void* binHandle = nullptr;
    void* ccuBinHandle = nullptr;
    bool hasReg = false;
    uint32_t binLen = 0U;
    DoubleListNode dllNode;
    CoreType coreType = kCoreTypeEnd;
    bool isStaticShape = false;
    uint32_t numBlocks = 0U;
    uint8_t staticWorkspaceSizes[NNOPBASE_WORKSPACE_STRUCT_SIZE];
    std::string kernelName;
    uint32_t multiKernelType = 0U;
    NnopbaseLoadBinInfoType loadBinInfoType = kDynamicInfo;
    std::unordered_map<uint64_t, MixRationParam> tilingKeyInfo;
    NnopbaseDfxInfo dfxInfo;
    size_t debugBufSize = 0U;
    uint32_t opParaSize = 0U;
    bool oomFlag = false;
    NnopbaseTaskRation taskRation = kRationEnd;
    std::vector<NnopbaseInitValueInfo> initValues;
    std::shared_ptr<MemsetOpInfo> memsetInfo = nullptr;
    std::shared_ptr<ExtraKernelDesc> extraKernelDesc = nullptr;
} NnopbaseBinInfo;

aclnnStatus NnopbaseMC2DynamicKernelRegister(const bool useCoreTypeMagic, NnopbaseBinInfo* binInfo);
aclnnStatus NnopbaseAclrtBinaryLoad(const bool useCoreTypeMagic, NnopbaseBinInfo* binInfo);

aclnnStatus NnopbaseReadKernelJsonFile(NnopbaseBinInfo* binInfo, const std::string& oppPath,
                                       const std::string& socVersion);
aclnnStatus NnopbaseReadBinFile(const NnopbaseChar* const binPath, const NnopbaseUChar** bin, uint32_t* binLen);
aclnnStatus NnopbaseKernelUnRegister(void** handle);
aclnnStatus NnopbaseReadJsonConfig(const std::string& binaryInfoPath, nlohmann::json& binaryInfoConfig);
aclnnStatus NnopbaseGetKernelJsonPath(const std::string& binPath, std::string& jsonPath);
NnopbaseChar* NnopbaseGetmmErrorMsg();

// for memset aux operator
aclnnStatus NnopbaseGenMemsetInfo(NnopbaseBinInfo* binInfo, const std::string& oppPath, const std::string& socVersion);
aclnnStatus NnopbaseLoadMemsetJson(std::shared_ptr<MemsetOpInfo>& memsetInfo);
aclnnStatus NnopbaseGetBinPath(const std::string& jsonPath, std::string& binPath);
aclnnStatus NnopbaseGetMemsetBinInfo(std::shared_ptr<MemsetOpInfo>& memsetInfo);
aclnnStatus NnopbaseReadMemsetJsonInfo(const std::string& memsetBasePath, nlohmann::json& memsetJsonInfo,
                                       std::shared_ptr<MemsetOpInfo>& memsetInfo, const size_t initNum,
                                       const bool loadFuncHandleByTilingKey);
aclnnStatus NnopbaseRegisterMemsetBin(std::shared_ptr<MemsetOpBinInfo>& binInfo, bool loadFuncHandleByTilingKey);

// getFunction Handle from binInfo
aclnnStatus GetFuncHandleByEntry(void* binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle);
aclnnStatus GetFuncHandleByKernelName(void* binHandle, const char* kernelName, aclrtFuncHandle* funcHandle);
static const std::map<std::string, NnopbaseTaskRation> TASK_RATION_MAP = {
    {"1:0", kRation10}, {"0:1", kRation01}, {"1:1", kRation11}, {"1:2", kRation12}};
static const std::map<std::string, CoreType> g_nnopbaseKernelTypeMap = {{"MIX", kMix},
                                                                        {"AiCore", kAicore},
                                                                        {"VectorCore", kVectorcore},
                                                                        {"MIX_AICORE", kMixAiCore},
                                                                        {"MIX_VECTOR_CORE", kMixAiv},
                                                                        {"MIX_AIC", kMix}};

static inline void NnopbaseMemsetInfoDestroy(std::shared_ptr<MemsetOpInfo>& memsetInfo)
{
    if ((memsetInfo->binInfo != nullptr) && (memsetInfo->binInfo->bin != nullptr)) {
        delete[] (memsetInfo->binInfo->bin);
        memsetInfo->binInfo->bin = nullptr;
    }
    NnopbaseKernelRunContextExt contextExt = memsetInfo->contextExt;
    FREE(contextExt.context);
    NnopbaseComputeNodeInfoExt nodeExt = contextExt.nodeExt;
    FREE(nodeExt.buf);
    nodeExt.bufLen = 0;
    if (memsetInfo->tilingParseContext != nullptr) {
        delete[] (memsetInfo->tilingParseContext);
        memsetInfo->tilingParseContext = nullptr;
    }
    if (memsetInfo->tilingParseContextValue != nullptr) {
        delete[] (memsetInfo->tilingParseContextValue);
        memsetInfo->tilingParseContextValue = nullptr;
    }
}

static inline void NnopbaseBinInfoDestroy(NnopbaseBinInfo** binInfo)
{
    if ((*binInfo)->bin != nullptr && (*binInfo)->loadBinInfoType != kStaticBinInfo) {
        delete[] ((*binInfo)->bin);
        (*binInfo)->bin = nullptr;
    }
    if ((*binInfo)->hasReg) {
        (void)NnopbaseKernelUnRegister(&((*binInfo)->binHandle));
    }
    if ((*binInfo)->memsetInfo != nullptr) {
        NnopbaseMemsetInfoDestroy((*binInfo)->memsetInfo);
    }
    (*binInfo)->initValues.clear();
    delete (*binInfo);
    *binInfo = nullptr;
}

static inline void NnopbaseBinInfoKeyCopy(BinInfoKey* dst, const BinInfoKey* src)
{
    dst->verbose = src->verbose;
    dst->hashKey = src->hashKey;
    dst->len = src->len;
    dst->bufLen = src->len;
}

static inline void NnopbaseBinInfoInit(NnopbaseBinInfo* binInfo)
{
    binInfo->bin = nullptr;
    binInfo->binHandle = nullptr;
    binInfo->ccuBinHandle = nullptr;
    binInfo->binLen = 0;
    binInfo->coreType = kCoreTypeEnd;
    binInfo->hasReg = false;
    binInfo->isStaticShape = false;
    binInfo->numBlocks = 0;
    auto vec = op::internal::PtrCastTo<gert::ContinuousVector>(binInfo->staticWorkspaceSizes);
    vec->Init(NNOPBASE_NORM_MAX_WORKSPACE_NUMS);
    binInfo->initValues.clear();
}

static inline aclnnStatus NnopbaseBinInfoSetOpBinInfoKey(NnopbaseBinInfo* binInfo, const NnopbaseUChar* const verbose,
                                                         const uint32_t len)
{
    NNOPBASE_ASSERT_TRUE_RETVAL(len > 0U);
    binInfo->binInfoKey.verbose = std::vector<NnopbaseUChar>(len, '\0');
    NNOPBASE_ASSERT_TRUE_RETVAL(memcpy_s(&(binInfo->binInfoKey.verbose[0U]), len, verbose, len) == EOK);
    binInfo->binInfoKey.len = len;
    binInfo->binInfoKey.bufLen = len;
    binInfo->binInfoKey.hashKey = NnopbaseHashBinary(verbose, len) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
    return OK;
}

#ifdef __cplusplus
}
#endif

namespace nnopbase {
std::string GetMagicFormBin(const bool useCoreTypeMagic, NnopbaseBinInfo* binInfo);
} // namespace nnopbase

#endif
