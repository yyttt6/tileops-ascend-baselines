/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_DFX_INTERNAL_H_
#define OP_DFX_INTERNAL_H_
#include <atomic>
#include "profiling/aprof_pub.h"
#include "op_dfx_util.h"
#include "op_cache_internal.h"
#include "opdev/op_dfx.h"
#include "launch_arg_info.h"
#include "bridge_pool.h"

namespace op {
namespace internal {
constexpr uint32_t OP_DFX_TASK_RATION_TWO = 2;
using AddrRestorer = std::tuple<const aclTensor*, void*>;
int OpGetTid();

uint32_t CollectProfilingStr(const char* s);

void PrepareBasicInfo(MsprofCompactInfo& compactInfo, MsprofGeTaskType taskType, uint64_t id, uint64_t summaryId);
void PrepareBasicInfo(MsprofCompactInfo& compactInfo, const TaskInfo& taskInfo, uint64_t id, uint64_t summaryId);

void PrepareAdditionInfo(MsprofAdditionalInfo& additionInfo);

struct CacheOpInfoBasic {
    uint32_t taskType;
    uint32_t numBlocks;
    uint64_t nodeId;
    uint64_t opType;
    uint64_t attrId{0};
    uint64_t reserve2{0};
    uint32_t opFlag;
    uint32_t tensorNum;
    MsrofTensorData tensorData[0];
};

void GetCacheOpInfoSwitch([[maybe_unused]] const aclrtStream& stream);
void ReportCacheOpInfo(const TaskInfo& taskInfo, OpArgContext* args, const uint64_t& opType,
                       const uint64_t& attrId = 0);
void ReportCacheOpInfoFromCache(const TaskInfo& taskInfo, void* tensorInfoLists, const uint32_t& numBlocks,
                                const ProfilingInfoId& profilingInfoId);
void ReportCacheOpInfoDSA(OpArgList& inputs, OpArgList& outputs, MsprofGeTaskType taskType);

void PrepareTensorInfo(const FVector<const aclTensor*>& tensors, MsprofTensorInfo& tensorInfo, uint64_t summaryId,
                       MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex);

void ReportAdditionInfo(const TaskInfo& taskInfo, uint64_t id, uint64_t summaryId);

void ReportNodeContextIdInfo(uint64_t summaryId);

void ChangeAclTenorAddrWithOffset(const aclTensor* tensor, std::vector<AddrRestorer>& record);
void RecoverAclTensorAddr(std::vector<AddrRestorer>& record);

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                       OpIOType ioType, std::vector<AddrRestorer>& record);

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<op::Tensor*>& opTensors,
                       OpIOType ioType);

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                       const std::vector<uint32_t>& tensorsIdxList, OpIOType ioType, std::vector<AddrRestorer>& record);

void DumpL2(const std::vector<const aclTensor*>& tensors, const OpLogInfo& opLogInfo, OpIOType ioType,
            aclrtStream stream);

void CacheTensorInfo(const FVector<const aclTensor*>& inTensors, const FVector<const aclTensor*>& outTensors);

void CacheDfxInfo(uint32_t numBlocks, const ProfilingInfoId& id, const TaskInfo& taskInfo, bool isMemSet);

int32_t RestoreTensorInfo(void* cacheTensorInfoLists, FVector<const aclTensor*>& inTensors,
                          FVector<const aclTensor*>& outTensors);

void DestoryTensorsCached(void* cacheTensorInfoLists);

constexpr int kMaxDurationInfoLen = 256;

extern SystemConfig systemConfig;

inline void GetProfilingTensors(OpArgList& argList, FVector<const aclTensor*>& tensors)
{
    TraitsAclTensor(tensors, argList);
}

void ReportAdditionInfo(FVector<const aclTensor*>& inTensors, FVector<const aclTensor*>& outTensors,
                        MsprofGeTaskType taskType, uint64_t summaryId);

void ReportAdditionInfo(FVector<const aclTensor*>& inTensors, FVector<const aclTensor*>& outTensors,
                        const TaskInfo& taskInfo, uint64_t summaryId);

inline void ReportAdditionInfo(OpArgList& inputs, OpArgList& outputs, MsprofGeTaskType taskType, uint64_t summaryId)
{
    FVector<const aclTensor*> inTensors;
    FVector<const aclTensor*> outTensors;
    GetProfilingTensors(inputs, inTensors);
    GetProfilingTensors(outputs, outTensors);
    ReportAdditionInfo(inTensors, outTensors, taskType, summaryId);
}
inline void ReportAdditionInfo(OpArgList& inputs, OpArgList& outputs, const TaskInfo& taskInfo, uint64_t summaryId)
{
    FVector<const aclTensor*> inTensors;
    FVector<const aclTensor*> outTensors;
    GetProfilingTensors(inputs, inTensors);
    GetProfilingTensors(outputs, outTensors);
    ReportAdditionInfo(inTensors, outTensors, taskInfo, summaryId);
}

void SummaryAttrArg([[maybe_unused]] size_t idx, OpArg& value, std::vector<AttrInfo>& attrInfos, std::string& attrStr);
void SummaryAttrArg([[maybe_unused]] size_t idx, OpArg& value, std::string& attrStr);

void ReportAttrInfo(std::string& attrStr, uint64_t summaryId);

inline void ReportAttrInfo(OpArgList& attrs, std::string& attrStr, std::vector<AttrInfo>& attrInfos)
{
    attrs.VisitByNoReturn(
        [&attrInfos, &attrStr](size_t idx, OpArg& elem) { SummaryAttrArg(idx, elem, attrInfos, attrStr); });
}

inline void PrepareL0DumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, OpArgList& tensors, OpIOType ioType,
                                std::vector<AddrRestorer>& record)
{
    std::vector<const aclTensor*> aclTensors;
    TraitsAclTensor(aclTensors, tensors);
    PrepareDumpTensor(dumpTensors, aclTensors, ioType, record);
}

void AddExceptionDumpOperatorInfo(const std::vector<Adx::TensorInfoV2>& dumpInTensors,
                                  const std::vector<Adx::TensorInfoV2>& dumpOutTensors,
                                  const std::vector<Adx::TensorInfoV2>& dumpWorkSpaceTensors,
                                  const OpLogInfo& opLogInfo, const aclrtStream stream);

void PrepareExceptionDumpInfo(OpArgContext* args, const OpLogInfo& opLogInfo, bool genPlaceholder, bool hasDevPtrArg_,
                              bool interCoreSync, std::vector<int32_t>& tensorOffset, const aclrtStream stream);

void PrepareExceptionDumpInfo(const std::vector<op::Tensor*>& in, const std::vector<op::Tensor*>& out,
                              const OpLogInfo& opLogInfo, const ExceptionDumpInfo& exceptionDumpInfo,
                              const aclrtStream stream);

aclnnStatus OverflowDumpProcess(OpArgContext* args, aclOpExecutor* executor, aclrtStream stream,
                                op::internal::OpLogInfo& logInfo);

void DumpL0(OpArgList& tensors, const OpLogInfo& opLogInfo, OpIOType ioType, aclrtStream stream);

void DumpL0(OpArgList& inputTensors, OpArgList& outputTensors, const OpLogInfo& opLogInfo, aclrtStream stream);

void OpCacheTid();

int32_t ProfilingCallBack(uint32_t type, VOID_PTR data, uint32_t len);

void PrepareTensorData(const FVector<const aclTensor*>& tensors, MsprofTensorInfo& tensorInfo, MsprofGeTensorType type,
                       uint32_t tensorNum, uint32_t baseIndex);

void PrepareL2DumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                         OpIOType ioType);

// OpProfilingSwitch 线程安全访问辅助函数 (仅针对 recordOpArgFlag)
bool GetOpProfilingRecordArgFlag();
void SetOpProfilingRecordArgFlag(bool value);

extern std::atomic<bool> g_aclnnDebugEnabled;

void InitAclnnDebugSwitch();
inline bool IsAclnnDebugEnabled() { return g_aclnnDebugEnabled.load(std::memory_order_relaxed); }

#ifdef __cplusplus
extern "C" {
#endif

bool CanUsePTACache(const char* api);

#ifdef __cplusplus
}
#endif
} // namespace internal
} // namespace op

#endif
