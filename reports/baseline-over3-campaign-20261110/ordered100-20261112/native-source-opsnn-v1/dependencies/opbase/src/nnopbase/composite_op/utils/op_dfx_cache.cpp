/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <string>
#include <vector>
#include <mutex>
#include "kernel_launcher.h"
#include "op_dfx_internal.h"
#include "op_cache_internal.h"
#include "opdev/op_def.h"
#include "opdev/op_cache.h"
#include "opdev/platform.h"
#include "mmpa/mmpa_api.h"

namespace op {
namespace internal {

struct TensorsCached {
    TensorsCached(const aclTensor* t, OpIOType type, [[maybe_unused]] OpExecCache* cache)
    {
        (void)aclGetStorageShape(t, &storageShape_, &storageShapeNum_);
        (void)aclGetDataType(t, &dateType_);
        (void)GetStorageFormat(t, &format_);
        ioType_ = type;
    }
    ~TensorsCached()
    {
        if (storageShape_ != nullptr) {
            delete[] storageShape_;
        }
        if (tensor_ == nullptr) {
            return;
        }
        if (tensor_->GetPlacement() == op::TensorPlacement::kOnHost) {
            auto addr = static_cast<uint8_t*>(tensor_->GetAddr());
            delete[] addr;
        }
        delete tensor_;
    }

    void GetStorageFormat(const aclTensor* tensor, Format* format) const
    {
        if (tensor == nullptr || format == nullptr) {
            return;
        }
        *format = tensor->GetStorageFormat();
    }

    Format GetProfilingFormat() const { return format_; }

    DataType GetProfilingDataType() const { return op::ToOpDataType(dateType_); }

    int64_t* GetProfilingShape() const { return storageShape_; }

    uint64_t GetProfilingDimNum() const { return storageShapeNum_; }

    int64_t* storageShape_{nullptr};
    uint64_t storageShapeNum_{0};
    aclDataType dateType_;
    Format format_;
    OpIOType ioType_;
    AddrRule addrRule_;
    Tensor* tensor_{nullptr};
};

int32_t RestoreTensorInfo(void* cacheTensorInfoLists, FVector<const TensorsCached*>& inTensors,
                          FVector<const TensorsCached*>& outTensors);

void PrepareTensorDataFromCache(const FVector<const TensorsCached*>& tensors, MsprofTensorInfo& tensorInfo,
                                MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex);

void PrepareTensorInfoFromCache(const FVector<const TensorsCached*>& tensors, MsprofTensorInfo& tensorInfo,
                                uint64_t summaryId, MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex);

void ReportAdditionInfo(FVector<const TensorsCached*>& inTensors, FVector<const TensorsCached*>& outTensors,
                        const TaskInfo& taskInfo, const op::internal::ProfilingInfoId& profilingInfoId);

int32_t RestoreDumpTensorAddr(TensorsCached* tensorCached, void* workspaceAddr, const std::vector<void*>& tensors);

void DestoryTensorsCached(void* cacheTensorInfoLists)
{
    if (cacheTensorInfoLists == nullptr) {
        return;
    }
    std::vector<TensorsCached*>* tensors = (std::vector<TensorsCached*>*)cacheTensorInfoLists;
    for (auto it : *tensors) {
        delete it;
    }
    delete tensors;
}

void CacheTensorInfo(const FVector<const aclTensor*>& inTensors, const FVector<const aclTensor*>& outTensors)
{
    OP_LOGD("CacheTensorInfo inTensors.size = %lu, outTensors.size = %lu", inTensors.size(), outTensors.size());
    OpExecCache* cache = GetOpCacheContext().GetOpCache();
    if (cache == nullptr) {
        return;
    }
    std::vector<const TensorsCached*>* cacheTensorInfoLists = new (std::nothrow) std::vector<const TensorsCached*>;
    if (cacheTensorInfoLists == nullptr) {
        return;
    }
    for (auto t : inTensors) {
        if (t == nullptr) {
            continue;
        }
        const TensorsCached* ti = new (std::nothrow) TensorsCached(t, OpInputType, cache);
        if (ti != nullptr) {
            cacheTensorInfoLists->push_back(ti);
        }
    }

    for (auto t : outTensors) {
        if (t == nullptr) {
            continue;
        }
        const TensorsCached* to = new (std::nothrow) TensorsCached(t, OpOutputType, cache);
        if (to != nullptr) {
            cacheTensorInfoLists->push_back(to);
        }
    }
    cache->SetCacheTensorInfo(cacheTensorInfoLists);
}

void CacheDfxInfo(uint32_t numBlocks, const ProfilingInfoId& id, const TaskInfo& taskInfo, bool isMemSet)
{
    OP_LOGD("CacheDfxInfo numBlocks = %u", numBlocks);
    OpExecCache* cache = GetOpCacheContext().GetOpCache();
    if (cache == nullptr) {
        return;
    }
    cache->SetBlockDim(numBlocks);
    cache->opExecCacheDfx_->SetProfilingInfoId(id);

    cache->opExecCacheDfx_->SetTaskInfo(taskInfo);
    if (!IsExceptionDumpEnable()) {
        return;
    }
    ExceptionDumpInfo exceptionDumpInfo;
    if (!isMemSet) {
        exceptionDumpInfo = GetThreadLocalContext().exceptionDumpInfo_;
    }
    cache->opExecCacheDfx_->SetExceptionDumpInfo(exceptionDumpInfo);
}

int32_t RestoreTensorInfo(void* infoLists, FVector<const TensorsCached*>& inTensors,
                          FVector<const TensorsCached*>& outTensors)
{
    CHECK_RET(infoLists != nullptr, -1);
    std::vector<TensorsCached*>* cacheTensorInfoLists = (std::vector<TensorsCached*>*)infoLists;
    OP_LOGD("RestoreTensorInfo cacheTensorInfoLists.size = %lu", cacheTensorInfoLists->size());
    for (auto it : *cacheTensorInfoLists) {
        if (it->ioType_ == OpInputType) {
            inTensors.push_back(it);
        }
        if (it->ioType_ == OpOutputType) {
            outTensors.push_back(it);
        }
    }
    OP_LOGD("RestoreTensorInfo inTensors.size = %lu, outTensors = %lu", inTensors.size(), outTensors.size());
    return 0;
}

void PrepareTensorDataFromCache(const FVector<const TensorsCached*>& tensors, MsprofTensorInfo& tensorInfo,
                                MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex)
{
    for (size_t i = 0; i < tensorNum; i++) {
        MsrofTensorData& tensorData = tensorInfo.tensorData[i];
        tensorData.tensorType = type;
        tensorData.format = tensors[i + baseIndex]->GetProfilingFormat();
        tensorData.dataType = tensors[i + baseIndex]->GetProfilingDataType();
        int64_t* shape = tensors[i + baseIndex]->GetProfilingShape();
        uint64_t dim = tensors[i + baseIndex]->GetProfilingDimNum();
        for (size_t j = 0; j < dim && j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            tensorData.shape[j] = shape[j];
        }
        for (size_t j = dim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            tensorData.shape[j] = 0;
        }
    }
}

void PrepareTensorInfoFromCache(const FVector<const TensorsCached*>& tensors, MsprofTensorInfo& tensorInfo,
                                uint64_t summaryId, MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex)
{
    tensorInfo.opName = summaryId;
    tensorInfo.tensorNum = tensorNum;
    PrepareTensorDataFromCache(tensors, tensorInfo, type, tensorNum, baseIndex);
}

static void ReportCachedTensorAdditionInfo(const FVector<const TensorsCached*>& tensors,
                                           MsprofAdditionalInfo& additionInfo, uint64_t summaryItemId,
                                           MsprofGeTensorType type)
{
    MsprofTensorInfo tensorInfo;
    uint32_t loop = tensors.size() / MSPROF_GE_TENSOR_DATA_NUM;
    uint32_t tail = tensors.size() % MSPROF_GE_TENSOR_DATA_NUM;
    for (uint32_t i = 0; i < loop; i++) {
        PrepareTensorInfoFromCache(tensors, tensorInfo, summaryItemId, type, MSPROF_GE_TENSOR_DATA_NUM,
                                   i * MSPROF_GE_TENSOR_DATA_NUM);
        OP_CHECK(memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &tensorInfo,
                          sizeof(MsprofTensorInfo)) == EOK,
                 OP_LOGW("Failed to memcpy tensor additional info."), return);
        MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
    }
    if (tail != 0) {
        PrepareTensorInfoFromCache(tensors, tensorInfo, summaryItemId, type, tail, loop * MSPROF_GE_TENSOR_DATA_NUM);
        OP_CHECK(memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &tensorInfo,
                          sizeof(MsprofTensorInfo)) == EOK,
                 OP_LOGW("Failed to memcpy tensor tail additional info."), return);
        MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
    }
}

void ReportAdditionInfo(FVector<const TensorsCached*>& inTensors, FVector<const TensorsCached*>& outTensors,
                        const TaskInfo& taskInfo, const op::internal::ProfilingInfoId& profilingInfoId)
{
    OP_LOGD("Entering function ReportAdditionInfo in cache input TaskInfo");
    if (MSPROF_ADDTIONAL_INFO_DATA_LENGTH != sizeof(MsprofTensorInfo)) {
        return;
    }

    MsprofAdditionalInfo additionInfo;
    MsprofCompactInfo compactInfo;

    PrepareBasicInfo(compactInfo, taskInfo, profilingInfoId.kernelLauncherId_, profilingInfoId.summaryItemId_);
    PrepareAdditionInfo(additionInfo);

    ReportCachedTensorAdditionInfo(inTensors, additionInfo, profilingInfoId.summaryItemId_,
                                   MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportCachedTensorAdditionInfo(outTensors, additionInfo, profilingInfoId.summaryItemId_,
                                   MSPROF_GE_TENSOR_TYPE_OUTPUT);
}

aclnnStatus DoReportAdditionInfo(void* infoLists, const TaskInfo& taskInfo,
                                 const op::internal::ProfilingInfoId& profilingInfoId)
{
    op::FVector<const TensorsCached*> inTensors;
    op::FVector<const TensorsCached*> outTensors;
    int res = RestoreTensorInfo(infoLists, inTensors, outTensors);
    CHECK_RET(res != -1, ACLNN_ERR_INNER);
    ReportAdditionInfo(inTensors, outTensors, taskInfo, profilingInfoId);
    return ACLNN_SUCCESS;
}

static void ReportCacheOpInfoTensorFromCache(uint8_t* dest, uint64_t& destOffset, const uint32_t& totalSize,
                                             const FVector<const TensorsCached*>& tensors, MsprofGeTensorType type)
{
    for (size_t i = 0; i < tensors.size(); i++) {
        // MsrofTensorData: tensorType/format/dataType/shape
        MsrofTensorData msTensor;
        msTensor.tensorType = static_cast<uint32_t>(type);
        msTensor.format = static_cast<uint32_t>(tensors[i]->GetProfilingFormat());
        msTensor.dataType = static_cast<uint32_t>(tensors[i]->GetProfilingDataType());
        int64_t* tensorShape = tensors[i]->GetProfilingShape();
        size_t dim = tensors[i]->GetProfilingDimNum();
        for (size_t j = 0; j < dim && j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            msTensor.shape[j] = tensorShape[j];
        }
        for (size_t j = dim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            msTensor.shape[j] = 0;
        }
        OP_CHECK(memcpy_s(dest + destOffset, totalSize - destOffset, &msTensor, sizeof(MsrofTensorData)) == EOK,
                 OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
                 throw std::runtime_error("aclGraph profiling memcpy runtime error."));
        destOffset += sizeof(MsrofTensorData);
        OP_LOGI("tensorIndex %zu, tensorType %u, format %u, dataType %u, shape(%u, %u, %u, %u, %u, %u, %u, %u)", i,
                msTensor.tensorType, msTensor.format, msTensor.dataType, msTensor.shape[0], msTensor.shape[1],
                msTensor.shape[2], msTensor.shape[3], msTensor.shape[4], msTensor.shape[5], msTensor.shape[6],
                msTensor.shape[7]);
    }
}

void ReportCacheOpInfoFromCache(const TaskInfo& taskInfo, void* tensorInfoLists, const uint32_t& numBlocks,
                                const ProfilingInfoId& profilingInfoId)
{
    OP_LOGI("Entering function ReportCacheOpInfoFromCache.");
    CacheOpInfoBasic cacheOpInfoBasic;
    // tensorNum
    op::FVector<const TensorsCached*> inTensors;
    op::FVector<const TensorsCached*> outTensors;
    OP_CHECK_NO_RETURN(RestoreTensorInfo(tensorInfoLists, inTensors, outTensors) == ACLNN_SUCCESS,
                       OP_LOGW("RestoreTensorInfo not ok."));
    uint32_t tensorNum = inTensors.size() + outTensors.size();
    size_t totalSize = sizeof(CacheOpInfoBasic) + (sizeof(MsrofTensorData) * tensorNum);
    void* infoPtr = op::internal::Allocate(totalSize);
    OP_CHECK(infoPtr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "infoPtr allocate failed."), throw std::bad_alloc());
    uint8_t* dest = static_cast<uint8_t*>(infoPtr);
    uint64_t destOffset = 0;
    // CacheOpInfoBasic: taskType/nodeId/opType/numBlocks/opFlag/tensorNum
    cacheOpInfoBasic.taskType = static_cast<uint32_t>(taskInfo.type);
    cacheOpInfoBasic.nodeId = profilingInfoId.summaryItemId_;
    cacheOpInfoBasic.opType = profilingInfoId.kernelLauncherId_;
    cacheOpInfoBasic.attrId = taskInfo.attrId;
    uint32_t currBlockDim = numBlocks;
    OP_LOGI("numBlocks is %u, taskInfo.ration is %u", currBlockDim, taskInfo.ration);
    if (taskInfo.ration == OP_DFX_TASK_RATION_TWO) {
        currBlockDim = ((currBlockDim & 0xFFFFU) | (0x20000));
    } else if (taskInfo.ration == 1) {
        currBlockDim = ((currBlockDim & 0xFFFFU) | (0x10000));
    }
    cacheOpInfoBasic.numBlocks = currBlockDim;
    OP_LOGI("numBlocks is %u after calculation", currBlockDim);
    cacheOpInfoBasic.opFlag = (static_cast<uint32_t>(taskInfo.execMode) &
                               static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32)) != 0 ?
                                  1 :
                                  0;
    cacheOpInfoBasic.tensorNum = tensorNum;
    OP_CHECK(memcpy_s(dest + destOffset, totalSize - destOffset, &cacheOpInfoBasic, sizeof(CacheOpInfoBasic)) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
             throw std::runtime_error("aclGraph profiling memcpy runtime error."));
    destOffset += sizeof(CacheOpInfoBasic);
    OP_LOGI("taskType %u, nodeId %zu, opType %zu, attrId %zu, numBlocks %u, opFlag %u, tensorNum %u",
            cacheOpInfoBasic.taskType, cacheOpInfoBasic.nodeId, cacheOpInfoBasic.opType, cacheOpInfoBasic.attrId,
            cacheOpInfoBasic.numBlocks, cacheOpInfoBasic.opFlag, cacheOpInfoBasic.tensorNum);
    ReportCacheOpInfoTensorFromCache(dest, destOffset, totalSize, inTensors, MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportCacheOpInfoTensorFromCache(dest, destOffset, totalSize, outTensors, MSPROF_GE_TENSOR_TYPE_OUTPUT);
    OP_CHECK_NO_RETURN(aclrtCacheLastTaskOpInfo(infoPtr, totalSize) == ACL_SUCCESS,
                       OP_LOGW("aclrtCacheLastTaskOpInfo failed"));
    op::internal::DeAllocate(infoPtr);
}

int32_t RestoreDumpTensorAddr(TensorsCached* tensorCached, void* workspaceAddr, const std::vector<void*>& tensors)
{
    if (tensorCached->tensor_ == nullptr) {
        OP_LOGW("RestoreDumpTensorAddr cached op tensor is nullptr.");
        return -1;
    }
    if (tensorCached->tensor_->GetPlacement() == op::TensorPlacement::kOnHost) {
        return 0;
    }
    void* newAddr;
    if (tensorCached->addrRule_.isWorkspace) {
        newAddr = PtrShift(workspaceAddr, tensorCached->addrRule_.workspaceOffset);
    } else {
        if (static_cast<size_t>(tensorCached->addrRule_.l2TensorInx) >= tensors.size()) {
            OP_LOGW("RestoreDumpTensorAddr cannot find L2 tensor, current idx %d, L2 tensor size %lu",
                    tensorCached->addrRule_.l2TensorInx, tensors.size());
            return -1;
        }
        newAddr = PtrShift(tensors[tensorCached->addrRule_.l2TensorInx], tensorCached->addrRule_.l2TensorOffset);
    }
    tensorCached->tensor_->MutableTensorData().SetAddr(newAddr, nullptr);
    return 0;
}

void DoExceptionDump(void* infoLists, void* workspaceAddr, const std::vector<void*>& tensors,
                     const ExceptionDumpInfo& dumpInfo, const aclrtStream stream)
{
    // memset dont need to dump
    if (dumpInfo.IsEmpty()) {
        return;
    }
    CHECK_RET(infoLists != nullptr, );
    std::vector<TensorsCached*>* cacheTensorInfoLists = (std::vector<TensorsCached*>*)infoLists;
    OP_LOGI("RestoreDumpTensorInfo cacheTensorInfoLists.size = %lu", cacheTensorInfoLists->size());
    std::vector<op::Tensor*> in;
    std::vector<op::Tensor*> out;
    for (auto it : *cacheTensorInfoLists) {
        auto ret = RestoreDumpTensorAddr(it, workspaceAddr, tensors);
        if (ret != 0) {
            OP_LOGW("RestoreDumpTensorAddr fail, stop exception dump.");
            return;
        }
        if (it->ioType_ == OpInputType) {
            in.push_back(it->tensor_);
        }
        if (it->ioType_ == OpOutputType) {
            out.push_back(it->tensor_);
        }
    }
    PrepareExceptionDumpInfo(in, out, GetThreadLocalContext().logInfo_, dumpInfo, stream);
}

} // namespace internal
} // namespace op
