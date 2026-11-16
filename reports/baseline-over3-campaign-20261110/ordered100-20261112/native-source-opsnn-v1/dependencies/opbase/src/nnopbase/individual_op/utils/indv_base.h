/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_COMMON_BASE_H_
#define INDV_COMMON_BASE_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#include <time.h>
#include <iostream>
#include <iomanip>
#include "indv_types.h"
#include "securec.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "individual_op_internal.h"
#include "individual_op_api.h"
#include "opdev/op_dfx.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr uint32_t NNOPBASE_NORM_DEF_IO_NUMS = 20U;
static constexpr int32_t NNOPBASE_NORM_DEF_TENSOR_NUMS = 100;
static constexpr size_t NNOPBASE_NORM_MAX_BIN_BUCKETS = 1024U;
static constexpr size_t NNOPBASE_SOC_VERSION_MAX = 50U;
static constexpr size_t NNOPBASE_VEB_KEY_LEN = 1024U;
// 算子genSimplifiedKey可写入的最大长度，取值须与composite_op侧MAX_CUSTOMIZED_SIMPLIFIED_KEY_LEN一致
static constexpr size_t NNOPBASE_CUS_KEY_LEN = 256U;
static constexpr uint32_t NNOPBASE_STATIC_VEB_KEY_LEN = 512U;

constexpr size_t NNOPBASE_BLOCK_NUM = 75U;
static constexpr size_t NNOPBASE_NORM_MAX_WORKSPACE_NUMS = 20U;
static constexpr size_t NNOPBASE_WORKSPACE_STRUCT_SIZE = sizeof(gert::ContinuousVector) +
                                                         NNOPBASE_NORM_MAX_WORKSPACE_NUMS * sizeof(size_t);

// 1字节在16进制占2位
static constexpr size_t NNOPBASE_BYTE_FOR_HEX = 2U;

enum NnopbaseTimeIdx {
    kAfterCreateExecutor = 0U,
    kGetWsStart,
    kMatchCacheStart,
    kMatchCacheEnd,
    kCreateCacheEnd,
    kAgingCacheEnd,
    kFindStaticEnd,
    kFindBinEnd,
    kTilingStart,
    kTilingEnd,
    kRunWithWsStart,
    kBeforeLaunch,
    kAfterLaunch,
    kRunWithWsEnd,
    kNum
};

enum NnopbaseCollectorTimeIdx {
    kCollectorInitStart = 0,
    kCollectorInitEnd,
    kGetBasePathEnd,
    kLoadTilingSoEnd,
    kLoadDebugKernelEnd,
    kLoadStaticKernelEnd,
    kLoadDynamicKernelEnd,
    kEnd
};

struct NnopbaseTimeStamp {
    struct timespec tp[NnopbaseTimeIdx::kNum];
};

typedef struct {
    void* overflowAddr;
    uint8_t deterministicLevel;
    NnopbaseUChar precision;
    bool enableArgsCache;
    bool enableTimeStamp;
    std::string implMode;
    bool enableDebugKernel;
} NnopbaseSysGlobalParams;

typedef struct {
    NnopbaseAnchorInstanceInfo* instStart;
    NnopbaseCompileTimeTensorDesc* inputTdStart;
    NnopbaseCompileTimeTensorDesc* outputTdStart;
    NnopbaseRuntimeAttrsDef* attrStart;
    NnopbaseAnchorInstanceInfo* outputInstStart;
    NnopbaseUChar* buf;
    size_t bufLen;
    NnopbaseComputeNodeInfo* node;
} NnopbaseComputeNodeInfoExt;

typedef struct {
    NnopbaseKernelRunContext* context;
    NnopbaseComputeNodeInfoExt nodeExt;
    uint32_t contextLen;
    bool hasPrepared;
    // customizedKey模式查找bin时会提前构造tilingContext，标记本次调用是否已构造，避免DoTiling重复构造
    bool hasBuilt;
} NnopbaseKernelRunContextExt;

typedef gert::TypedContinuousVector<size_t> NnopbaseWorkspaceSizes;

typedef struct {
    uint8_t tilingData[NNOPBASE_TILING_DATA_STRUCT_SIZE] = {};
    uint8_t workspacesSizes[NNOPBASE_WORKSPACE_STRUCT_SIZE] = {};
    uint64_t tilingKey = 0UL;
    uint32_t numBlocks = 0U;
    bool needAtomic = false;
    uint32_t scheMode = 0U;
    uint32_t aicpuNumBlocks = 0U;
    uint32_t dynUbufSize = 0U;
    std::vector<uint8_t> staticTilingData;
} NnopbaseTilingInfo;

typedef struct {
    void* workspaces[NNOPBASE_NORM_MAX_WORKSPACE_NUMS];
    size_t num;
    uint64_t length;
    uint32_t workspaceArgsOffset[NNOPBASE_NORM_MAX_WORKSPACE_NUMS];
    std::vector<GertTensor> workspaceTensor;
} NnopbaseWorkSpaces;

typedef struct {
    uint32_t num;    // 每个原型实例的tensor个数
    uint32_t cfgNum; // 0:optional 1:required 2:dynamic
    uint32_t startIndex;
    bool isInput;
    bool isDynamic;
    std::vector<uint8_t> scalarValue;
    int32_t scalarIndex = -1;
    int32_t refIdx = -1;
    const char* name = nullptr;
    bool isDisable;
    bool ignoreCont = false;
    const aclTensor* tensor = nullptr;
    const aclTensorList* tensorList = nullptr;
    const aclIntArray* intArray = nullptr;
    const aclBoolArray* boolArray = nullptr;
    const aclFloatArray* floatArray = nullptr;
    const aclScalar* scalar = nullptr;
    const aclScalarList* scalarList = nullptr;
    ge::DataType scalarDtype = ge::DT_UNDEFINED;
    int32_t srcIndex = -1;
} NnopbaseParamInstance;

typedef struct {
    std::vector<NnopbaseParamInstance> instances;
    uint32_t count;            // 原型个数
    uint32_t emptyNum;         // 空tensor个数
    uint32_t activeInputCount; // active的index个数
} NnopbaseParamDesc;

typedef struct {
    GertTensor rt2Tensor;
    bool isNull = true; // false 表示传入的aclTensor不为nullptr
    bool isRequired = false;
    bool isOptional = false;
    bool valueDepend = false;
    uint32_t argsOffset = 0U;
} NnopbaseTensor;

struct NnopbaseTensorList : public op::Object {
    const aclTensor** tensors = nullptr;
    uint64_t size = 0U;
};

struct NnopbaseUnContTensors {
    std::vector<const aclTensor*> tensors;
    std::vector<size_t> idxs;
    op::FVector<uint64_t> workspaceOffsets;
    NnopbaseTensorList tensorList;

    std::vector<size_t> refIdxs;
    std::vector<const aclTensor*> refUnContTensors;
    NnopbaseTensorList refUnContTensorList;
    std::vector<aclTensor*> refContTensors;
    aclTensorList* refContTensorList = nullptr;
};

typedef struct {
    std::vector<NnopbaseTensor> extTensors;
    NnopbaseUnContTensors unContiguousTensors;
    NnopbaseParamDesc paramDescs;
    uint32_t usedNum = 0U;
    uint32_t arrayLen = 0U;
    uint32_t requiredCnt = 0U;
    uint32_t nonDynamicCnt = 0U;
    uint32_t num = 0U; // tensor总个数
    bool hasDynamic = false;
    uint16_t hostInputNum = 0U;
    uint32_t expectIndex = 0U;
    size_t hostInputSize = 0U;
    uint32_t dynamicNum = 0U; // DYNAMIC输入/输出原型个数
    uint32_t dynamicCnt = 0U; // DYNAMIC输入/输出中tensor总数
    size_t dynamicSize = 0U;
    size_t outPutShapeSize = 0U;
    size_t outPutShapeArgsOffset = 0U;
    std::map<uint32_t, aclTensor*> outPutShapeMap;
} NnopbaseTensors;

struct NnopbaseAttr {
    NnopbaseAttrAddr addr;
    NnopbaseAttrDtype dtype;
};

typedef struct {
    NnopbaseAttr* attrs = nullptr;
    size_t num = 0U;
    size_t totalSize = 0U;
    size_t totalDataLen = 0U;
} NnopbaseAttrs;

static inline NnopbaseUChar* NnopbaseAppend1Byte(NnopbaseUChar* buf, NnopbaseUChar src)
{
    *buf = src;
    return buf + 1;
}

static inline NnopbaseUChar* NnopbaseAppend4Byte(void* buf, uint32_t src)
{
    uint32_t* dst = (uint32_t*)buf;
    *dst = src;
    return (uint8_t*)buf + sizeof(src);
}

static inline NnopbaseUChar* NnopbaseAppend8Byte(NnopbaseUChar* buf, uint64_t src)
{
    uint64_t* p = op::internal::PtrCastTo<uint64_t>(buf);
    *p = src;
    return buf + sizeof(uint64_t);
}

static inline void NnopbasePrintHex(const uint8_t* const p, const size_t num, std::stringstream& ss)
{
    for (size_t i = 0U; i < num; ++i) {
        ss << "0x" << std::setfill('0') << std::setw(NNOPBASE_BYTE_FOR_HEX) << std::hex << +p[i] << ' ';
    }
}

#ifdef __cplusplus
}
namespace nnopbase {
std::string ToStr(const NnopbaseAttrDtype dtype);
std::string ToStr(const NnopbaseAttrs& attrs);
std::string ToStr(const NnopbaseAttr& attr);
std::string ToStr(const GertTensor& tensor);
} // namespace nnopbase
#endif
#endif
