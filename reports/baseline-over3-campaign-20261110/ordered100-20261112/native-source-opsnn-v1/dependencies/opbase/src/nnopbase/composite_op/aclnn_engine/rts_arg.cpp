/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_arg.h"
#include <memory>
#include <new>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <iostream>
#include <sstream>

#include "kernel_arg.h"
#include "launcher_ctx.h"

#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "opdev/op_cache.h"
#include "op_dfx_internal.h"
#include "op_cache_internal.h"
#include "kernel_utils.h"
#include "bridge_dfx.h"
#include "dump/adump_api.h"

#define RT_PROTECT_EXTERNAL 1
#include "runtime/rt_external.h"

constexpr int DeviceHolder = 0;
constexpr int TilingHolder = 1;
constexpr int HostHolder = 2;
constexpr int Int64Btyes = 8;
constexpr size_t HostValueAlignment = 32;
constexpr size_t ExceptionDumpHead = 2;
constexpr uint64_t InputOffsetWithFFTS = 4294967296;
constexpr int MAX_CACHE_TILING_SIZE = 65536;
constexpr int64_t ASSERT_WORKSPACE_FLAG = 4;
constexpr int64_t ASSERT_WORKSPACE_SIZE = ASSERT_WORKSPACE_FLAG << 56;
constexpr size_t KERNEL_ATTRS_SIZE_THREE = 3;
constexpr size_t KERNEL_ATTRS_SIZE_FOUR = 4;

namespace op::internal {
bool IsNeedOverflowStatusAddr()
{
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_SATURATION;
    const aclError ret = aclrtGetDeviceSatMode(&floatOverflowMode);
    OP_LOGI("Get DeviceSatMode ret = %d, mode = %d.", ret, static_cast<int>(floatOverflowMode));
    if (floatOverflowMode == ACL_RT_OVERFLOW_MODE_SATURATION) {
        return true;
    } else {
        return false;
    }
}

static int HostDataType(uint32_t offset, const rtArgs_t& rtArg)
{
    if (rtArg.hasTiling != 0 && offset == rtArg.tilingAddrOffset) {
        return TilingHolder;
    }
    aclrtPlaceHolderInfo* placeHolderInfo = rtArg.placeHolderInfoPtr;
    if (placeHolderInfo != nullptr && rtArg.placeHolderInfoNum != 0) {
        for (uint32_t i = 0; i < rtArg.placeHolderInfoNum; i++) {
            if (offset == placeHolderInfo[i].addrOffset) {
                return HostHolder;
            }
        }
    }
    return DeviceHolder;
}

void PrintHostDataSize(const rtArgs_t& rtArg)
{
    aclrtPlaceHolderInfo* placeHolderInfo = rtArg.placeHolderInfoPtr;
    OP_CHECK(placeHolderInfo != nullptr, OP_LOGW("placeHolderInfo is nullptr"), return);
    OP_CHECK(rtArg.placeHolderInfoNum != 0, OP_LOGW("placeHolderInfoNum is 0"), return);

    uint32_t hostDataNum = (rtArg.hasTiling != 0) ? rtArg.placeHolderInfoNum - 1 : rtArg.placeHolderInfoNum;
    OP_LOGI("hasTiling: %u, host data number: %u", rtArg.hasTiling, hostDataNum);
    uint32_t dataLen = 0;
    for (uint16_t i = 0; i < hostDataNum; i++) {
        if (placeHolderInfo[i].addrOffset == rtArg.tilingAddrOffset) {
            continue;
        }
        if (i < hostDataNum - 1) {
            dataLen = placeHolderInfo[i + 1].dataOffset - placeHolderInfo[i].dataOffset;
        } else {
            if (rtArg.tilingDataOffset > placeHolderInfo[i].dataOffset) {
                dataLen = rtArg.tilingDataOffset - placeHolderInfo[i].dataOffset;
            } else {
                dataLen = rtArg.argsSize - placeHolderInfo[i].dataOffset;
            }
        }
        std::stringstream ss;
        ss << "HostInput[" << i << "], addrOffset: " << placeHolderInfo[i].addrOffset << ", Aligned Size: " << dataLen
           << ", Data(uint64_t): ";
        for (size_t j = 0; j < (dataLen / Int64Btyes); j++) {
            ss << "0x" << std::hex
               << *PtrCastTo<uint64_t>(PtrShift(rtArg.args, placeHolderInfo[i].dataOffset + j * Int64Btyes)) << " ";
        }
        SplitDataAndPrint("PrintHostDataSize", ss.str());
    }
}

void PrintTilingData(const rtArgs_t& rtArg)
{
    if (rtArg.hasTiling == 0) {
        OP_LOGI("Op does not have tiling");
        return;
    }
    if (rtArg.tilingDataOffset >= rtArg.argsSize) {
        OP_LOGW("tilingDataOffset %u >= argsSize %u, hasTiling %u, skip print.", rtArg.tilingDataOffset, rtArg.argsSize,
                rtArg.hasTiling);
        return;
    }
    uint32_t tilingLen = 0;
    aclrtPlaceHolderInfo* placeHolderInfo = rtArg.placeHolderInfoPtr;
    if (placeHolderInfo != nullptr && rtArg.placeHolderInfoNum > 1 &&
        placeHolderInfo[0].dataOffset > rtArg.tilingDataOffset) {
        tilingLen = placeHolderInfo[0].dataOffset - rtArg.tilingDataOffset;
    } else {
        tilingLen = rtArg.argsSize - rtArg.tilingDataOffset;
    }

    std::stringstream ss;
    ss << "Aligned TilingData Size: " << tilingLen << ", data(uint32_t): ";
    for (uint32_t i = 0; i < tilingLen; i += sizeof(uint32_t)) {
        ss << *PtrCastTo<uint32_t>(PtrShift(rtArg.args, rtArg.tilingDataOffset + i)) << " ";
    }
    SplitDataAndPrint("PrintTilingData", ss.str());
}

int PrintRtArg(const rtArgs_t& rtArg)
{
    if (rtArg.hasTiling != 0 && rtArg.placeHolderInfoNum == 0) {
        OP_LOGW("Has tiling data but placeHolderInfoNum is 0");
    }
    uint32_t hostDataNum = (rtArg.hasTiling != 0 && rtArg.placeHolderInfoNum > 0) ? rtArg.placeHolderInfoNum - 1 :
                                                                                    rtArg.placeHolderInfoNum;
    OP_LOGD("opType: %s, argsSize: %u, placeHolderInfoNum: %u, hostDataNum: %u, hasTiling %d, tilingAddrOffset: %u, "
            "tilingDataOffset: %u",
            op::internal::GetThreadLocalContext().logInfo_.l0Name, rtArg.argsSize, rtArg.placeHolderInfoNum,
            hostDataNum, rtArg.hasTiling, rtArg.tilingAddrOffset, rtArg.tilingDataOffset);
    OP_CHECK(rtArg.args != nullptr, OP_LOGW("args is null, no need to print."), return -1);

    aclrtPlaceHolderInfo* placeHolderInfo = rtArg.placeHolderInfoPtr;
    uint32_t hostDataOffset = static_cast<uint32_t>(0xffffffff);
    if (placeHolderInfo != nullptr && rtArg.placeHolderInfoNum != 0) {
        for (uint32_t i = 0; i < rtArg.placeHolderInfoNum; i++) {
            hostDataOffset = std::min(hostDataOffset, placeHolderInfo[i].dataOffset);
        }
    } else {
        OP_LOGW("No placeholder info, use argsSize as loop bound.");
        hostDataOffset = rtArg.argsSize;
    }
    if (hostDataOffset > rtArg.argsSize) {
        OP_LOGW("hostDataOffset %u exceeds argsSize %u, clamping to argsSize.", hostDataOffset, rtArg.argsSize);
        hostDataOffset = rtArg.argsSize;
    }
    void** addr = (void**)(rtArg.args);
    int hosti = 0;
    std::string rtArgInfo;
    for (uint32_t i = 0; i < (hostDataOffset / sizeof(void*)); i++) {
        int hostType = HostDataType(i * sizeof(void*), rtArg);
        std::stringstream ss;
        if (hostType == DeviceHolder) {
            ss << addr[i] << " ";
        } else if (hostType == TilingHolder) {
            ss << "tiling ";
        } else {
            ss << "host" << hosti++ << " ";
        }
        rtArgInfo.append(ss.str());
    }
    SplitDataAndPrint("PrintRtArg", rtArgInfo);
    PrintHostDataSize(rtArg);
    PrintTilingData(rtArg);
    return 0;
}

int PrintExceptionDumpInfo(void* dump, size_t num)
{
    int64_t* exception = PtrCastTo<int64_t>(dump);
    std::string dumpInfo = "";
    for (size_t i = 0; i < num; i++) {
        dumpInfo += (std::to_string(exception[i]) + " ");
    }
    SplitDataAndPrint("PrintExceptionDumpInfo", dumpInfo);
    return 0;
}

int PrintAICErrorDFXInfo(const void* dfxInfoAddr, const size_t argNum, const size_t dataSize)
{
    std::string tensorSizeInfo = "tensor size(uint64_t):";
    size_t tensorSizeInfoSize = argNum * sizeof(uint64_t);
    const uint64_t* tensorSizeInfoPtr = PtrCastTo<uint64_t>(dfxInfoAddr);
    for (uint16_t i = 0; i < tensorSizeInfoSize; i += sizeof(uint64_t)) {
        tensorSizeInfo += (" " + std::to_string(*PtrCastTo<uint64_t>(PtrShift(tensorSizeInfoPtr, i))));
    }
    std::string shapeInfo = "shape info(uint64_t):";
    size_t shapeInfoLen = dataSize - tensorSizeInfoSize;
    const uint64_t* shapeInfoPtr = PtrCastTo<uint64_t>(PtrShift(dfxInfoAddr, tensorSizeInfoSize));
    for (uint16_t i = 0; i < shapeInfoLen; i += sizeof(uint64_t)) {
        shapeInfo += (" " + std::to_string(*PtrCastTo<uint64_t>(PtrShift(shapeInfoPtr, i))));
    }
    std::stringstream ss;
    ss << tensorSizeInfo << ", " << shapeInfo;
    SplitDataAndPrint("PrintAICErrorDFXInfo", ss.str());
    return 0;
}

constexpr size_t TILING_AND_OVERFLOW = 2;
constexpr size_t ONLY_OVERFLOW = 1;
thread_local std::vector<aclrtPlaceHolderInfo> RtsArg::placeHolderInfo_{MAX_HOST_INFO_NUM};

RtsArg::RtsArg(bool hasFftsAddr, const LaunchArgInfo& argInfo, ExpandableRtsArgBuffer* rtsArgBuffer)
    : hasFftsAddr_(hasFftsAddr), argInfo_(argInfo), rtArg_({}), rtsArgBuffer_(rtsArgBuffer)
{
    // tilingDataLen 从 buffer 获取（LaunchArgInfo 已移除 tilingData_/tilingDataLen_）
    size_t tilingDataLen = rtsArgBuffer_->GetTilingDataPtr()->data_size_;
    // +2 for tiling addr and overflow_addr
    if (tilingDataLen == 0) {
        argNum_ = (hasFftsAddr ? 1 : 0) + argInfo.GetDevArgNum() + argInfo.GetHostArgNum() + ONLY_OVERFLOW;
        rtArg_.hasTiling = false;
        OP_LOGD("Kernel has no tiling: hasFftsAddr %d, devArgNum %zu, hostArgNum %zu, argNum %zu", hasFftsAddr,
                argInfo.GetDevArgNum(), argInfo.GetHostArgNum(), argNum_);
    } else {
        argNum_ = (hasFftsAddr ? 1 : 0) + argInfo.GetDevArgNum() + argInfo.GetHostArgNum() + TILING_AND_OVERFLOW;
        rtArg_.hasTiling = true;
        OP_LOGD("Kernel has tiling: hasFftsAddr %d, devArgNum %zu, hostArgNum %zu, argNum %zu", hasFftsAddr,
                argInfo.GetDevArgNum(), argInfo.GetHostArgNum(), argNum_);
    }

    // 阶段1: 确保容量（可能触发扩容，但此时没有已填数据，无需刷新）
    // 1a. 确保 launch_arg 容量并设置已用大小
    size_t requiredLaunchArgSize = argNum_ * sizeof(void*);
    OP_CHECK(rtsArgBuffer_->EnsureLaunchArgCapacity(requiredLaunchArgSize) == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER, "failed to ensure launch arg capacity, required %zu.", requiredLaunchArgSize),
             throw std::bad_alloc());
    rtsArgBuffer_->SetLaunchArgSize(requiredLaunchArgSize);

    // 1b. 确保 tiling_host_data 容量
    size_t alignedTilingDataLen = AlignSize(tilingDataLen, HOST_VALUE_ALIGNMENT);
    OP_CHECK(rtsArgBuffer_->UpdateTilingDataSize(alignedTilingDataLen) == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER, "failed to ensure tiling host data size %zu.", alignedTilingDataLen),
             throw std::bad_alloc());

    // 阶段2: 从最终基地址设置指针（一次性，无需检测扩容）
    rtArg_.args = static_cast<void*>(static_cast<uint8_t*>(rtsArgBuffer_->GetTilingDataAddr()) - requiredLaunchArgSize);
    hostAddr_ = static_cast<void**>(rtArg_.args);
    hostAddrOffset_ = 0;

    // 清零对齐填充
    if (alignedTilingDataLen > tilingDataLen) {
        OP_CHECK((memset_s(PtrShift(rtsArgBuffer_->GetTilingDataAddr(), tilingDataLen),
                           alignedTilingDataLen - tilingDataLen, 0, alignedTilingDataLen - tilingDataLen) == EOK),
                 OP_LOGW("Failed to memset."),
                 ;);
    }
    placeHolderInfo_.clear();
}

RtsArg::~RtsArg() { rtsArgBuffer_->ResetTilingHostDataCursor(); }

void RtsArg::ReportExceptionDumpInfo() const
{
#ifdef RT_PROTECT_EXTERNAL
    if (exceptionDumpAddr_ == nullptr) {
        return;
    }
    rtArgsSizeInfo sizeInfo;
    sizeInfo.infoAddr = exceptionDumpAddr_;
    sizeInfo.atomicIndex = exceptionDumpIndex_;
    if (rtSetExceptionExtInfo(&sizeInfo) != ACL_SUCCESS) {
        OP_LOGW("rtSetExceptionExtInfo failed.");
    }
    OP_LOGI("%d", PrintExceptionDumpInfo(exceptionDumpAddr_, argInfo_.GetTensorNum() + ExceptionDumpHead));
#endif
}

aclnnStatus RtsArg::LaunchKernel(aclrtStream stream, const KernelLaunchConfig& launchCfg)
{
    OP_LOGI("Launch kernel engine type: %d, numBlocks: %u, scheduleMode: %u, blockDimOffset: %u, dynUBufSize: %u, "
            "funcHandle: %p, print arg result: %d",
            static_cast<int>(launchCfg.engineType), launchCfg.numBlocks, launchCfg.schemMode, launchCfg.blockDimOffset,
            launchCfg.dynUBufSize, launchCfg.funcHandle, PrintRtArg(rtArg_));

    std::vector<aclrtLaunchKernelAttr> kernelAttrs;
    kernelAttrs.reserve(KERNEL_ATTRS_SIZE_THREE);
    kernelAttrs.emplace_back();
    kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    kernelAttrs.back().value.schemMode = launchCfg.schemMode;

    if (launchCfg.engineType == LaunchKernelEngineType::NO_VECTOR_CORE) {
        kernelAttrs.emplace_back();
        kernelAttrs.back().id = static_cast<aclrtLaunchKernelAttrId>(LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
        kernelAttrs.back().value.localMemorySize = launchCfg.dynUBufSize;
    } else if (launchCfg.engineType == LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC ||
               launchCfg.engineType == LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIV) {
        kernelAttrs.emplace_back();
        kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
        kernelAttrs.back().value.blockDimOffset = launchCfg.blockDimOffset;
        kernelAttrs.emplace_back();
        kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
        kernelAttrs.back().value.engineType = (launchCfg.engineType == LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC) ?
                                                  ACL_RT_ENGINE_TYPE_AIC :
                                                  ACL_RT_ENGINE_TYPE_AIV;
    }

    aclrtLaunchKernelCfg aclrtLaunchCfg;
    aclrtLaunchCfg.attrs = kernelAttrs.data();
    aclrtLaunchCfg.numAttrs = kernelAttrs.size();

    aclError rc = aclrtLaunchKernelWithHostArgs(launchCfg.funcHandle, launchCfg.numBlocks, stream, &aclrtLaunchCfg,
                                                rtArg_.args, rtArg_.argsSize, rtArg_.placeHolderInfoPtr,
                                                rtArg_.placeHolderInfoNum);
    OP_CHECK(
        rc != ACL_ERROR_RT_INVALID_HANDLE,
        OP_LOGW("aclrtLaunchKernelWithHostArgs return %d, need to update function handle", ACL_ERROR_RT_INVALID_HANDLE),
        return ACLNN_ERR_INNER_RUNTIME_INVALID_HANDLE);
    CHECK_COND(rc == ACL_SUCCESS, ACLNN_ERR_RUNTIME_ERROR, "aclrtLaunchKernelWithHostArgs failed, return: %d", rc);
    return ACLNN_SUCCESS;
}

void RtsArg::AppendExceptionDumpAddr(bool assertFlag)
{
    if (!assertFlag && !op::internal::IsNeedL0ExceptionDump()) {
        return;
    }
    auto& allArg = argInfo_.GetAllArgInfo();
    const size_t inputNum = argInfo_.GetArgSize();
    if (inputNum == 0) {
        OP_LOGI("AppendExceptionDumpAddr input arg num is 0, no need dump.");
        return;
    }
    auto dumpSize = argInfo_.GetTensorNum() + ExceptionDumpHead;
    exceptionDumpAddr_ = Adx::AdumpGetSizeInfoAddr(dumpSize, exceptionDumpIndex_);
    if (exceptionDumpAddr_ == nullptr) {
        OP_LOGW("AdumpGetSizeInfoAddr get SizeInfoAddr error.");
        return;
    }
    OP_LOGI("AdumpGetSizeInfoAddr space is %zu, index is %d, exceptionDumpAddr is 0x%p", dumpSize, exceptionDumpIndex_,
            exceptionDumpAddr_);
    uint64_t* sizeInfoAddr = PtrCastTo<uint64_t>(exceptionDumpAddr_);
    // atmoic index
    *sizeInfoAddr = static_cast<uint64_t>(exceptionDumpIndex_);
    sizeInfoAddr++;
    // input offset and input nums
    *sizeInfoAddr = (hasFftsAddr_ ? InputOffsetWithFFTS : 0) + static_cast<uint64_t>(argInfo_.GetTensorNum());
    sizeInfoAddr++;
    // input byte sizes
    int64_t* inputAddr = PtrCastTo<int64_t>(sizeInfoAddr);
    size_t firstWorkspaceIdx = argInfo_.GetFirstWorkspaceIdx();
    for (size_t i = 0; i < inputNum; i++) {
        if (allArg[i].tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_ARG && allArg[i].devAddr_.tensor != nullptr) {
            int64_t tensorSize = op::CalcShapeBytes(allArg[i].devAddr_.tensor->GetStorageShape().GetShapeSize(),
                                                    allArg[i].devAddr_.tensor->GetDataType());
            // assert 场景，第一个workspace的size高8位置4
            if (assertFlag && firstWorkspaceIdx == i) {
                tensorSize += ASSERT_WORKSPACE_SIZE;
            }
            *inputAddr = tensorSize;
        } else if (allArg[i].tag_ == LaunchArgInfo::ArgAddr::ArgTag::HOST_ARG && allArg[i].devAddr_.tensor != nullptr) {
            *inputAddr = allArg[i].hostAddr_.hostDataLen;
        } else if (allArg[i].tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_PTR_ARG) {
            // 高8位是特殊标识，低56位表示二级指针中一共有多少个tensor
            uint64_t devicePtrMark = static_cast<uint64_t>(2) << 56;
            auto tensors = allArg[i].devPtrAddr_.tensors;
            *inputAddr++ = devicePtrMark + tensors->Size();
            for (size_t j = 0; j < tensors->Size(); j++) {
                ((*tensors)[j] == nullptr) ?
                    (*inputAddr++ = 0) :
                    (*inputAddr++ = op::CalcShapeBytes((*tensors)[j]->GetStorageShape().GetShapeSize(),
                                                       (*tensors)[j]->GetDataType()));
            }
            inputAddr--; // because it has ++ in loop last
        } else {
            *inputAddr = 0;
        }
        inputAddr++;
    }
}

aclnnStatus RtsArg::AppendFftsAddr()
{
    void* modeAddr = nullptr;
    aclError rc = aclrtGetHardwareSyncAddr(&modeAddr);
    OP_CHECK(rc == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtGetHardwareSyncAddr failed: %d", rc),
             return ACLNN_ERR_RUNTIME_ERROR);
    *hostAddr_ = modeAddr;
    hostAddr_++;
    hostAddrOffset_ += PTR_SIZE;

    return ACL_SUCCESS;
}

void RtsArg::RefreshRtArgsAddr()
{
    void* oldArgs = rtArg_.args;
    rtArg_.args = static_cast<void*>(static_cast<uint8_t*>(rtsArgBuffer_->GetTilingDataAddr()) -
                                     rtsArgBuffer_->GetLaunchArgSize());
    hostAddr_ = PtrCastTo<void*>(static_cast<uint8_t*>(rtArg_.args) + hostAddrOffset_);
    OP_LOGD("RefreshRtArgsAddr: args from %p to %p, hostAddrOffset %zu", oldArgs, rtArg_.args, hostAddrOffset_);
}

aclnnStatus RtsArg::AppendHostArg(void* hostData, size_t hostDataSize)
{
    bool isEmptyData = false;
    if (hostData == nullptr || hostDataSize == 0) {
        OP_LOGI("host data is null. %zu", hostDataSize);
        // reserve one block of device memory for null host data.
        hostDataSize = HOST_VALUE_ALIGNMENT;
        isEmptyData = true;
    }

    size_t savedGen = rtsArgBuffer_->GetGeneration();
    size_t savedOffset = rtsArgBuffer_->GetTilingHostDataSize();

    size_t alignedHostDataSize = AlignSize(hostDataSize, HOST_VALUE_ALIGNMENT);
    OP_CHECK(rtsArgBuffer_->SeekTilingHostData(alignedHostDataSize) == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER, "failed to seek buffer, len %zu.", alignedHostDataSize), return ACLNN_ERR_INNER);

    if (rtsArgBuffer_->GetGeneration() != savedGen) {
        RefreshRtArgsAddr();
    }
    void* hostDataAddr = static_cast<uint8_t*>(rtsArgBuffer_->GetTilingDataAddr()) + savedOffset;

    if (!isEmptyData) {
        OP_CHECK(memcpy_s(hostDataAddr, hostDataSize, hostData, hostDataSize) == EOK, OP_LOGW("Failed to memcpy."), ;);
        if (alignedHostDataSize > hostDataSize) {
            OP_CHECK((memset_s(static_cast<uint8_t*>(hostDataAddr) + hostDataSize, alignedHostDataSize - hostDataSize,
                               0, alignedHostDataSize - hostDataSize) == EOK),
                     OP_LOGW("Failed to memset."),
                     ;);
        }
    } else {
        OP_CHECK((memset_s(hostDataAddr, hostDataSize, 0, hostDataSize) == EOK), OP_LOGW("Failed to memset."), ;);
    }

    placeHolderInfo_.emplace_back(aclrtPlaceHolderInfo{
        static_cast<decltype(rtArg_.placeHolderInfoPtr->addrOffset)>(hostAddrOffset_),
        static_cast<decltype(rtArg_.placeHolderInfoPtr->dataOffset)>(PtrOffset(rtArg_.args, hostDataAddr))});
    tensorOffset_.push_back(hostAddrOffset_ / PTR_SIZE);
    hostAddr_++;
    hostAddrOffset_ += PTR_SIZE;
    return ACLNN_SUCCESS;
}

aclnnStatus RtsArg::AppendDevicePtrArg(const aclTensorList* tensors, size_t dataSize)
{
    // ptr_offset / dimNum 1 / dim0...n / dimNum 1 / dim0...n / ... / ptr1 / ptr2 / ...
    size_t hostDataSize = dataSize + tensors->Size() * PTR_SIZE;
    size_t alignedHostDataSize = AlignSize(hostDataSize, HOST_VALUE_ALIGNMENT);

    size_t savedGen = rtsArgBuffer_->GetGeneration();
    size_t savedOffset = rtsArgBuffer_->GetTilingHostDataSize();

    OP_CHECK(rtsArgBuffer_->SeekTilingHostData(alignedHostDataSize) == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER, "failed to seek buffer, len %zu.", alignedHostDataSize), return ACLNN_ERR_INNER);

    if (rtsArgBuffer_->GetGeneration() != savedGen) {
        RefreshRtArgsAddr();
    }

    uint8_t* hostDataAddr = static_cast<uint8_t*>(rtsArgBuffer_->GetTilingDataAddr()) + savedOffset;
    int64_t* shapeInfoPtr = PtrCastTo<int64_t>(hostDataAddr);
    int64_t* deviceAddrPtr = PtrCastTo<int64_t>(hostDataAddr + dataSize);

    *shapeInfoPtr++ = dataSize;
    for (size_t i = 0; i < tensors->Size(); i++) {
        if ((*tensors)[i] == nullptr) {
            *shapeInfoPtr++ = (static_cast<int64_t>(1) << DEV_PTR_DIM_SHIFT_BIT) + static_cast<int64_t>(0);
            tensorOffset_.push_back(PtrOffset(rtArg_.args, deviceAddrPtr) / PTR_SIZE);
            void** p = PtrCastTo<void*>(PtrCastTo<void>(deviceAddrPtr));
            *p = nullptr;
            deviceAddrPtr++;
            continue;
        }
        const auto& shape = (*tensors)[i]->GetStorageShape();
        size_t dim = shape.GetDimNum();
        *shapeInfoPtr++ = (static_cast<int64_t>(1) << DEV_PTR_DIM_SHIFT_BIT) + dim; // 低32位填维度大小, 高32位固定填1
        for (size_t j = 0; j < dim; j++) {
            *shapeInfoPtr++ = shape.GetDim(j);
        }
        tensorOffset_.push_back(PtrOffset(rtArg_.args, deviceAddrPtr) / PTR_SIZE);
        *deviceAddrPtr++ = reinterpret_cast<int64_t>((*tensors)[i]->GetData());
    }
    if (alignedHostDataSize > hostDataSize) {
        OP_CHECK((memset_s(hostDataAddr + hostDataSize, alignedHostDataSize - hostDataSize, 0,
                           alignedHostDataSize - hostDataSize) == EOK),
                 OP_LOGW("Failed to memset."),
                 ;);
    }

    placeHolderInfo_.emplace_back(aclrtPlaceHolderInfo{
        static_cast<decltype(rtArg_.placeHolderInfoPtr->addrOffset)>(hostAddrOffset_),
        static_cast<decltype(rtArg_.placeHolderInfoPtr->dataOffset)>(PtrOffset(rtArg_.args, hostDataAddr))});
    hostAddr_++;
    hostAddrOffset_ += PTR_SIZE;
    return ACLNN_SUCCESS;
}

aclnnStatus RtsArg::FinalizeArg()
{
    if (rtArg_.hasTiling != 0) {
        rtArg_.tilingAddrOffset = static_cast<uint32_t>(hostAddrOffset_);
        rtArg_.tilingDataOffset = static_cast<uint32_t>(PtrOffset(rtArg_.args, rtsArgBuffer_->GetTilingDataAddr()));
        placeHolderInfo_.emplace_back(aclrtPlaceHolderInfo{rtArg_.tilingAddrOffset, rtArg_.tilingDataOffset});
        hostAddr_++;
        hostAddrOffset_ += PTR_SIZE;
    } else {
        rtArg_.tilingAddrOffset = 0;
        rtArg_.tilingDataOffset = 0;
    }

    // append overflow addr
    AppendOverflowStatusAddr();
    rtArg_.placeHolderInfoPtr = placeHolderInfo_.data();
    rtArg_.placeHolderInfoNum = static_cast<uint32_t>(placeHolderInfo_.size());
    rtArg_.argsSize = PtrOffset(rtArg_.args, rtsArgBuffer_->GetTilingHostDataCurEndAddr());

    // 打印RtsArg全部信息
    OP_LOGI("TilingData ptr %p, tiling data data %p, tiling data size %zu, tiling data capacity %zu, "
            "launch arg size %zu, launch arg capacity %zu, aligned tiling data len %zu, host data size %zu, "
            "tiling host dataSize %zu, tiling host data capacity %zu, rtArg_.args %p, hostAddr_ %p, "
            "args size %u, tiling data addr %p, tiling host data end addr %p",
            rtsArgBuffer_->GetTilingDataPtr(), rtsArgBuffer_->GetTilingDataPtr()->data_,
            rtsArgBuffer_->GetTilingDataPtr()->data_size_, rtsArgBuffer_->GetTilingDataPtr()->capacity_,
            rtsArgBuffer_->GetLaunchArgSize(), rtsArgBuffer_->GetLaunchArgCapacity(),
            rtsArgBuffer_->GetAlignedTilingDataSize(), rtsArgBuffer_->GetHostDataSize(),
            rtsArgBuffer_->GetTilingHostDataSize(), rtsArgBuffer_->GetTilingHostDataCapacity(), rtArg_.args, hostAddr_,
            rtArg_.argsSize, rtsArgBuffer_->GetTilingDataAddr(), rtsArgBuffer_->GetTilingHostDataCurEndAddr());

    return ACLNN_SUCCESS;
}

aclnnStatus RtsArg::FillArgs(bool assertFlag)
{
    if (hasFftsAddr_) {
        OP_CHECK(AppendFftsAddr() == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "rtsArg fillArgs appendFftsAddr failed."),
                 return ACLNN_ERR_INNER);
    }
    auto& allArg = argInfo_.GetAllArgInfo();
    for (size_t i = 0; i < argInfo_.GetArgSize(); i++) {
        const auto& elem = allArg[i];
        if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_ARG) {
            AppendArg(elem.devAddr_.devAddr);
        } else if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_PTR_ARG) {
            OP_CHECK(AppendDevicePtrArg(elem.devPtrAddr_.tensors, elem.devPtrAddr_.ptrListLen) == ACLNN_SUCCESS,
                     OP_LOGE(ACLNN_ERR_INNER, "Append device ptr arg fail"), return ACLNN_ERR_INNER);
        } else {
            OP_CHECK(AppendHostArg(elem.hostAddr_.hostAddr, elem.hostAddr_.hostDataLen) == ACLNN_SUCCESS,
                     OP_LOGE(ACLNN_ERR_INNER, "Append host arg fail"), return ACLNN_ERR_INNER);
        }
    }
    OP_CHECK(FinalizeArg() == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "rtsArg fillArgs finalizeArg failed."),
             return ACLNN_ERR_INNER);
    AppendExceptionDumpAddr(assertFlag);
    if (IsExceptionDumpEnable()) {
        GetThreadLocalContext().exceptionDumpInfo_.rtsArgs_ = rtArg_.args;
        GetThreadLocalContext().exceptionDumpInfo_.rtsArgsSize_ = rtArg_.argsSize;
    }
    return ACLNN_SUCCESS;
}

void AddArgInfoToCache(OpExecCache* cache, LaunchArgCache::ArgInfo* argInfo, const LaunchArgInfo& launchArgInfo,
                       bool hasFftsAddr, const ExpandableRtsArgBuffer* rtsArgBuffer)
{
    size_t argIdx = 0;
    if (hasFftsAddr) {
        void* pFfsAddr = cache->AddLaunchData(sizeof(void*));
        OP_CHECK(pFfsAddr != nullptr, OP_LOGD("cache can't addLaunchData about fftsAddr, cache is invalid."), return);
        argInfo[argIdx++].type = LaunchArgCache::FFTS_ADDR;
    }
    auto& allArg = launchArgInfo.GetAllArgInfo();
    for (size_t i = 0; i < launchArgInfo.GetArgSize(); i++) {
        const auto& elem = allArg[i];
        if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_ARG) {
            if (elem.devAddr_.tensor != nullptr) {
                OP_CHECK((cache->AddLaunchTensor(elem.devAddr_.tensor, sizeof(void*)) != nullptr),
                         OP_LOGD("cache can't addLaunchTensor about devArgAddr, cache is invalid."), return);
                argInfo[argIdx++].type = LaunchArgCache::DEV_ADDR;
                OP_LOGD("Add cache launch tensor: %p, addr: %p", elem.devAddr_.tensor, elem.devAddr_.devAddr);
            } else {
                void** p = PtrCastTo<void*>(cache->AddLaunchData(sizeof(void*)));
                OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about devArgAddr, cache is invalid."),
                         return);
                *p = nullptr;
                argInfo[argIdx++].type = LaunchArgCache::DEV_ADDR;
                OP_LOGD("Add cache launch tensor: nullptr");
            }
        } else if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_PTR_ARG) {
            void** p = PtrCastTo<void*>(cache->AddLaunchData(sizeof(void*)));
            OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about devPtrArgAddr, cache is invalid."), return);
            *p = nullptr;
            argInfo[argIdx++].type = LaunchArgCache::DEV_PTR_ADDR;
            OP_LOGD("Add cache dev ptr");
        } else {
            void** p = PtrCastTo<void*>(cache->AddLaunchData(sizeof(void*)));
            OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about hostDataAddr, cache is invalid."), return);
            *p = nullptr;
            argInfo[argIdx++].type = LaunchArgCache::HOST_DATA;
            OP_LOGD("Add cache host data addr");
            // set host datalen later
        }
    }
    size_t additionalAddr = 2; // tilingdata addr and overflow addr
    void* pAdditionalAddr = cache->AddLaunchData(sizeof(void*) * additionalAddr);
    OP_CHECK(pAdditionalAddr != nullptr, OP_LOGD("cache can't addLaunchData about additionalAddr, cache is invalid."),
             return);
    LaunchArgCache::ArgInfo* tilingInfo = &argInfo[argIdx];
    argInfo[argIdx++].type = LaunchArgCache::TILING_DATA;
    argInfo[argIdx++].type = LaunchArgCache::OVERFLOW_ADDR;
    argIdx = hasFftsAddr ? 1 : 0;
    for (size_t i = 0; i < launchArgInfo.GetArgSize(); i++) {
        const auto& elem = allArg[i];
        if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_ARG) {
            argIdx++;
        } else if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::HOST_ARG) {
            void* hostData = elem.hostAddr_.hostAddr;
            size_t hostLen = elem.hostAddr_.hostDataLen;
            bool isEmptyData = false;
            if (hostData == nullptr || hostLen == 0) {
                hostLen = HostValueAlignment;
                isEmptyData = true;
            }
            size_t alignLen = AlignSize(hostLen, HostValueAlignment);
            void* p = cache->AddLaunchData(alignLen);
            OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about hostArgAddr, cache is invalid."), return);
            if (!isEmptyData) {
                OP_CHECK(memcpy_s(p, hostLen, hostData, hostLen) == EOK, OP_LOGW("Failed to memcpy."), ;);
                if (alignLen > hostLen) {
                    OP_CHECK((memset_s(PtrShift(p, hostLen), alignLen - hostLen, 0, alignLen - hostLen) == EOK),
                             OP_LOGW("Failed to memset."),
                             ;);
                }
            } else {
                OP_CHECK((memset_s(p, alignLen, 0, alignLen) == EOK), OP_LOGW("Failed to memset."), ;);
            }
            argInfo[argIdx++].dataLen = alignLen;
            OP_LOGD("Add cache host data content. size: %zu", alignLen);
        } else if (elem.tag_ == LaunchArgInfo::ArgAddr::ArgTag::DEVICE_PTR_ARG) {
            auto tensors = elem.devPtrAddr_.tensors;
            size_t dataSize = elem.devPtrAddr_.ptrListLen;
            int64_t* p = static_cast<int64_t*>(cache->AddLaunchData(dataSize));
            OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about devPtrArgAddr, cache is invalid."), return);
            *p++ = dataSize;
            for (size_t i = 0; i < tensors->Size(); i++) {
                if ((*tensors)[i] == nullptr) {
                    *p++ = (static_cast<int64_t>(1) << RtsArg::DEV_PTR_DIM_SHIFT_BIT) + static_cast<int64_t>(0);
                    continue;
                }
                const auto& shape = (*tensors)[i]->GetStorageShape();
                size_t dim = shape.GetDimNum();
                *p++ = (static_cast<int64_t>(1) << RtsArg::DEV_PTR_DIM_SHIFT_BIT) + dim;
                for (size_t j = 0; j < dim; j++) {
                    *p++ = shape.GetDim(j);
                }
            }
            for (size_t i = 0; i < tensors->Size(); i++) {
                if ((*tensors)[i] != nullptr) {
                    OP_CHECK((cache->AddLaunchTensor((*tensors)[i], sizeof(void*)) != nullptr),
                             OP_LOGD("cache can't addLaunchTensor about devPtrArgAddr, cache is invalid."), return);
                } else {
                    void** cacheData = PtrCastTo<void*>(cache->AddLaunchData(sizeof(void*)));
                    OP_CHECK(cacheData != nullptr,
                             OP_LOGD("cache can't addLaunchData about devPtrArgAddr, cache is invalid."), return);
                    *cacheData = nullptr;
                }
            }
            size_t dataLen = dataSize + tensors->Size() * sizeof(void*);
            size_t alignLen = AlignSize(dataLen, RtsArg::HOST_VALUE_ALIGNMENT);
            OP_CHECK(cache->AddLaunchData(alignLen - dataLen) != nullptr,
                     OP_LOGD("cache can't addLaunchData about hostAlignAddr, cache is invalid."), return);
            argInfo[argIdx++].dataLen = alignLen;
            OP_LOGD("Add cache device ptr, size: %zu", alignLen);
        }
    }
    TilingData* tilingData = rtsArgBuffer->GetTilingDataPtr();
    size_t tilingDataLen = tilingData->data_size_;
    size_t aligntilingDataLen = AlignSize(tilingDataLen, HostValueAlignment);
    void* p = cache->AddLaunchData(aligntilingDataLen);
    OP_CHECK(p != nullptr, OP_LOGD("cache can't addLaunchData about aligntilingDataAddr, cache is invalid."), return);
    OP_CHECK(memcpy_s(p, tilingDataLen, tilingData->data_, tilingDataLen) == EOK, OP_LOGW("Failed to memcpy."), ;);
    if (aligntilingDataLen > tilingDataLen) {
        OP_CHECK((memset_s(PtrShift(p, tilingDataLen), aligntilingDataLen - tilingDataLen, 0,
                           aligntilingDataLen - tilingDataLen) == EOK),
                 OP_LOGW("Failed to memset."),
                 ;);
    }
    tilingInfo->dataLen = aligntilingDataLen;
    OP_LOGD("Add cache tiling data content. size: %zu", aligntilingDataLen);
}

void RtsArg::AddExceptionDumpDataToCache(const LaunchArgInfo& argInfo, OpExecCache* cache,
                                         LaunchArgCache* launchCache) const
{
    if (exceptionDumpAddr_ == nullptr) {
        OP_LOGI("exception dump addr is null, skip cache.");
        return;
    }
    auto dumpArgNum = argInfo.GetTensorNum() + ExceptionDumpHead;
    auto exceptionDumpDataLen = dumpArgNum * Int64Btyes;
    void* exceptionAddr = cache->AddLaunchData(exceptionDumpDataLen);
    if (exceptionAddr != nullptr) {
        OP_CHECK(memcpy_s(exceptionAddr, exceptionDumpDataLen, exceptionDumpAddr_, exceptionDumpDataLen) == EOK,
                 OP_LOGW("Failed to memcpy exception dump data to cache."), return);
        launchCache->SetExceptionArgNum(dumpArgNum);
    }
}

void RtsArg::AddDFXInfoDumpDataToCache(const LaunchArgInfo& argInfo, OpExecCache* cache,
                                       LaunchArgCache* launchCache) const
{
    if (!IsArgExceptionDumpEnable()) {
        return;
    }
    const void* dfxInfoDumpAddr = argInfo.GetDFXInfoDumpAddr();
    uint32_t dfxInfoElemCount = argInfo.GetDFXInfoDumpElemCount();
    if (dfxInfoDumpAddr == nullptr || dfxInfoElemCount == 0) {
        OP_LOGI("DFX info dump addr: %p, dump elem count: %u, skip cache.", dfxInfoDumpAddr, dfxInfoElemCount);
        return;
    }
    void* dfxInfoCacheAddr = cache->AddLaunchData(dfxInfoElemCount * sizeof(uint64_t));
    OP_CHECK(dfxInfoCacheAddr != nullptr, OP_LOGW("Fail to add data in cache!"), return);
    OP_CHECK(memcpy_s(dfxInfoCacheAddr, dfxInfoElemCount * sizeof(uint64_t), dfxInfoDumpAddr,
                      dfxInfoElemCount * sizeof(uint64_t)) == EOK,
             OP_LOGW("Failed to memcpy DFX info dump data to cache."), return);
    launchCache->SetDFXInfoCacheElemCount(dfxInfoElemCount);
    launchCache->SetDFXInfoOffsetInTilingData(argInfo.GetDFXInfoOffsetInTilingData());
    launchCache->SetLaunchArgNum(argInfo.GetArgSize());

    OP_LOGI("cache addr: %p, DFX info offset in tiling data: %zu, print dfx info in cache: %d", dfxInfoCacheAddr,
            argInfo.GetDFXInfoOffsetInTilingData(),
            PrintAICErrorDFXInfo(dfxInfoCacheAddr, argInfo.GetArgSize(), dfxInfoElemCount * sizeof(uint64_t)));
}

LaunchArgCache* RtsArg::DumpToCache()
{
    OpExecCache* cache = GetOpCacheContext().GetOpCache();
    if (cache == nullptr) {
        OP_LOGD("no op cache in context");
        return nullptr;
    }
    OP_CHECK(!(cache->CanUse()), OP_LOGI("OpExecCache has been can used, cant dump to cache."), return nullptr);
    size_t tilingDataLen = rtsArgBuffer_->GetTilingDataPtr()->data_size_;
    if (!cache->IsOpCacheValid() || tilingDataLen == 0 || tilingDataLen >= MAX_CACHE_TILING_SIZE) {
        OP_LOGI("hash key is zero, or don't has tiling, or tiling size bigger than cache limit, skip cache.");
        cache->MarkOpCacheInvalid();
        return nullptr;
    }
    size_t offset;
    size_t cap = 0;
    cache->NewLaunchCache(&offset, &cap, LaunchArgCache::RunFromCache);
    OP_LOGD("New launch cache. offset: %zu, cap: %zu", offset, cap);
    size_t argInfoLen = argNum_ * sizeof(LaunchArgCache::ArgInfo);
    LaunchArgCache* launchCache = PtrCastTo<LaunchArgCache>(cache->AddLaunchData(sizeof(LaunchArgCache) + argInfoLen));
    OP_CHECK(launchCache != nullptr, OP_LOGD("cache can't addLaunchData in dumpToCache, cache is invalid."),
             return nullptr);
    launchCache->SetArgNum(argNum_);
    launchCache->SetExceptionArgNum(0);
    launchCache->SetDFXInfoCacheElemCount(0);
    launchCache->SetRtsApiType(LaunchArgCache::RTS_OLD);
    launchCache->SetOpType(op::internal::GetThreadLocalContext().logInfo_.l0Name);
    LaunchArgCache::ArgInfo* argInfo = launchCache->GetArgInfo();
    AddArgInfoToCache(cache, argInfo, argInfo_, hasFftsAddr_, rtsArgBuffer_);
    AddExceptionDumpDataToCache(argInfo_, cache, launchCache);
    AddDFXInfoDumpDataToCache(argInfo_, cache, launchCache);
    return launchCache;
}

void ReportRTSException(const LaunchArgCache* launchCache, void* cacheException)
{
#ifdef RT_PROTECT_EXTERNAL
    size_t dumpArgNum = launchCache->GetExceptionArgNum();
    if (dumpArgNum == 0) {
        return;
    }
    uint32_t curIndex = 0;
    void* newException = Adx::AdumpGetSizeInfoAddr(dumpArgNum, curIndex);
    if (newException == nullptr) {
        OP_LOGW("AdumpGetSizeInfoAddr get SizeInfoAddr error.");
        return;
    }
    auto exceptionDumpDataLen = dumpArgNum * Int64Btyes;
    OP_CHECK(memcpy_s(newException, exceptionDumpDataLen, cacheException, exceptionDumpDataLen) == EOK,
             OP_LOGW("Failed to memcpy."), return);
    *PtrCastTo<uint64_t>(newException) = static_cast<uint64_t>(curIndex);
    rtArgsSizeInfo sizeInfo;
    sizeInfo.infoAddr = newException;
    sizeInfo.atomicIndex = curIndex;
    if (rtSetExceptionExtInfo(&sizeInfo) != ACL_SUCCESS) {
        OP_LOGW("rtSetExceptionExtInfo failed.");
    }
    OP_LOGI("%d", PrintExceptionDumpInfo(newException, dumpArgNum));
#endif
}

static void UpdateDFXInfoDumpAndTilingData(rtArgs_t& rtArg, const LaunchArgCache* launchCache, void* dfxInfoCache)
{
    uint32_t dumpElemCount = launchCache->GetDFXInfoCacheElemCount();
    OP_CHECK(dumpElemCount != 0, OP_LOGW("DFX info elem count is 0, no need to update"), return);
    uint64_t dfxInfoDumpIndex = 0;
    void* newDFXInfoDumpAddr = Adx::AdumpGetDFXInfoAddrForDynamic(dumpElemCount, dfxInfoDumpIndex);
    OP_CHECK(newDFXInfoDumpAddr != nullptr,
             OP_LOGW("AdumpGetDFXInfoAddrForDynamic get address failed, request space: %u", dumpElemCount), return);
    OP_CHECK(memcpy_s(newDFXInfoDumpAddr, dumpElemCount * sizeof(uint64_t), dfxInfoCache,
                      dumpElemCount * sizeof(uint64_t)) == EOK,
             OP_LOGW("Failed to memcpy."), return);
    size_t dfxInfoOffset = launchCache->GetDFXInfoOffsetInTilingData();
    *PtrCastTo<uint64_t>(PtrShift(rtArg.args, rtArg.tilingDataOffset + dfxInfoOffset)) = dfxInfoDumpIndex;
    uint32_t* atomicIndexU32Type = PtrCastTo<uint32_t>(&dfxInfoDumpIndex);
    OP_LOGI("dump request space: %u, atomic index: %lu(hex: 0x%lX, uint32_t: %u %u), "
            "DFX info offset in tiling data: %zu, print dfx info dump: %d",
            dumpElemCount, dfxInfoDumpIndex, dfxInfoDumpIndex, atomicIndexU32Type[0], atomicIndexU32Type[1],
            dfxInfoOffset,
            PrintAICErrorDFXInfo(newDFXInfoDumpAddr, launchCache->GetLaunchArgNum(), dumpElemCount * sizeof(uint64_t)));
}

aclnnStatus LaunchArgCache::UpdateFunctionHandle(KernelLaunchConfig& launchCfg)
{
    aclrtFuncHandle funcHandle;
    if (launchCfg.isFatBin) {
        CHECK_COND(aclrtBinaryGetFunctionByEntry(launchCfg.binHandle, launchCfg.tilingKey, &funcHandle) == ACL_SUCCESS,
                   ACLNN_ERR_RUNTIME_ERROR, "aclrtBinaryGetFunctionByEntry failed, bin handle: %p, tiling key: %lu",
                   launchCfg.binHandle, launchCfg.tilingKey);
        OP_LOGI("Get function handle by tiling key [%lu] successfully, function handle: %p", launchCfg.tilingKey,
                funcHandle);
    } else {
        CHECK_COND(
            aclrtBinaryGetFunction(launchCfg.binHandle, launchCfg.kernelNameOfNoFatBin, &funcHandle) == ACL_SUCCESS,
            ACLNN_ERR_RUNTIME_ERROR, "aclrtBinaryGetFunction failed, kernel name: %s", launchCfg.kernelNameOfNoFatBin);
        OP_LOGI("Get function handle by kernel name [%s] successfully, function handle: %p",
                launchCfg.kernelNameOfNoFatBin, funcHandle);
    }
    launchCfg.funcHandle = funcHandle;
    return ACLNN_SUCCESS;
}

aclnnStatus LaunchArgCache::LaunchKernelFromCache(aclrtStream stream, rtArgs_t& rtArg, KernelLaunchConfig& launchCfg)
{
    OP_LOGD("Launch kernel engine type: %d, numBlocks: %u, scheduleMode: %u, blockDimOffset: %u, dynUBufSize: %u, "
            "funcHandle: %p",
            static_cast<int>(launchCfg.engineType), launchCfg.numBlocks, launchCfg.schemMode, launchCfg.blockDimOffset,
            launchCfg.dynUBufSize, launchCfg.funcHandle);

    std::vector<aclrtLaunchKernelAttr> kernelAttrs;
    kernelAttrs.reserve(KERNEL_ATTRS_SIZE_FOUR);
    kernelAttrs.emplace_back();
    kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    kernelAttrs.back().value.schemMode = launchCfg.schemMode;
    kernelAttrs.emplace_back();
    kernelAttrs.back().id = static_cast<aclrtLaunchKernelAttrId>(LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
    kernelAttrs.back().value.localMemorySize = launchCfg.dynUBufSize;
    if (launchCfg.engineType == LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC) {
        kernelAttrs.emplace_back();
        kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
        kernelAttrs.back().value.blockDimOffset = launchCfg.blockDimOffset;
        kernelAttrs.emplace_back();
        kernelAttrs.back().id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
        kernelAttrs.back().value.engineType = ACL_RT_ENGINE_TYPE_AIC;
    }

    aclrtLaunchKernelCfg aclrtLaunchCfg;
    aclrtLaunchCfg.attrs = kernelAttrs.data();
    aclrtLaunchCfg.numAttrs = kernelAttrs.size();

    aclError rc = aclrtLaunchKernelWithHostArgs(launchCfg.funcHandle, launchCfg.numBlocks, stream, &aclrtLaunchCfg,
                                                rtArg.args, rtArg.argsSize, rtArg.placeHolderInfoPtr,
                                                rtArg.placeHolderInfoNum);
    if (rc == ACL_ERROR_RT_INVALID_HANDLE) {
        OP_LOGW("aclrtLaunchKernelWithHostArgs return %d, need to update function handle", ACL_ERROR_RT_INVALID_HANDLE);
        OP_CHECK(UpdateFunctionHandle(launchCfg) == ACL_SUCCESS,
                 OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Update function handle failed"), return ACLNN_ERR_RUNTIME_ERROR);
        rc = aclrtLaunchKernelWithHostArgs(launchCfg.funcHandle, launchCfg.numBlocks, stream, &aclrtLaunchCfg,
                                           rtArg.args, rtArg.argsSize, rtArg.placeHolderInfoPtr,
                                           rtArg.placeHolderInfoNum);
    }
    OP_CHECK(rc == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtLaunchKernelWithHostArgs failed: %d", rc),
             return ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

aclnnStatus LaunchArgCache::RunFromCache(aclrtStream stream, void* cache)
{
    if (cache == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Null cache, unable to run");
        return ACLNN_ERR_INNER;
    }

    LaunchArgCache* launchCache = PtrCastTo<LaunchArgCache>(cache);
    op::internal::GetThreadLocalContext().logInfo_.l0Name = launchCache->GetOpType();

    LaunchArgCache::ArgInfo* argInfo = launchCache->GetArgInfo();
    void* rawArg = launchCache->GetRawRtsArg();
    size_t argNum = launchCache->GetArgInfoNum();
    size_t devAddrLen = argNum * sizeof(void*);
    size_t currHostDataLen = 0;
    RtsArg::placeHolderInfo_.clear();
    std::vector<aclrtPlaceHolderInfo>& placeHolderInfo = RtsArg::placeHolderInfo_;
    OP_CHECK(launchCache->GetRtsApiType() == LaunchArgCache::RTS_OLD,
             OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cache only support old rts, type is %u.",
                     static_cast<uint32_t>(launchCache->GetRtsApiType())),
             return ACLNN_ERR_RUNTIME_ERROR);
    rtArgs_t rtArg;
    rtArg.args = rawArg;
    rtArg.hasTiling = true;
    rtArg.argsSize = launchCache->GetRawArgSize();

    if (IsExceptionDumpEnable()) {
        op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgs_ = rtArg.args;
        op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgsSize_ = rtArg.argsSize;
    }

    for (size_t i = 0; i < argNum; i++) {
        if (argInfo[i].type == LaunchArgCache::FFTS_ADDR) {
            void** p = PtrCastTo<void*>(PtrShift(rawArg, sizeof(void*) * i));
            aclError rc = aclrtGetHardwareSyncAddr(p);
            OP_CHECK(rc == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtGetHardwareSyncAddr failed: %d", rc),
                     return ACLNN_ERR_RUNTIME_ERROR);
        } else if (argInfo[i].type == LaunchArgCache::HOST_DATA) {
            placeHolderInfo.emplace_back(aclrtPlaceHolderInfo{
                static_cast<decltype(rtArg.placeHolderInfoPtr->addrOffset)>(sizeof(void*) * i),
                static_cast<decltype(rtArg.placeHolderInfoPtr->dataOffset)>(devAddrLen + currHostDataLen)});
            currHostDataLen += argInfo[i].dataLen;
        } else if (argInfo[i].type == LaunchArgCache::TILING_DATA) {
            rtArg.tilingAddrOffset = sizeof(void*) * i;
            rtArg.tilingDataOffset = devAddrLen + currHostDataLen;
            currHostDataLen += argInfo[i].dataLen;
            placeHolderInfo.emplace_back(aclrtPlaceHolderInfo{rtArg.tilingAddrOffset, rtArg.tilingDataOffset});
        } else if (argInfo[i].type == LaunchArgCache::OVERFLOW_ADDR) {
            void* overflowAddr = nullptr;
            static bool needOverflowAddr = IsNeedOverflowStatusAddr();
            if (needOverflowAddr) {
                aclError rc = aclrtCtxGetFloatOverflowAddr(&overflowAddr);
                OP_CHECK(rc == ACL_SUCCESS,
                         OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtCtxGetFloatOverflowAddr failed: %d", rc),
                         return ACLNN_ERR_RUNTIME_ERROR);
            }
            void** p = PtrCastTo<void*>(PtrShift(rawArg, sizeof(void*) * i));
            *p = overflowAddr;
        } else if (argInfo[i].type == LaunchArgCache::DEV_PTR_ADDR) {
            placeHolderInfo.emplace_back(aclrtPlaceHolderInfo{
                static_cast<decltype(rtArg.placeHolderInfoPtr->addrOffset)>(sizeof(void*) * i),
                static_cast<decltype(rtArg.placeHolderInfoPtr->dataOffset)>(devAddrLen + currHostDataLen)});
            currHostDataLen += argInfo[i].dataLen;
        } else {
            continue;
        }
    }
    rtArg.placeHolderInfoPtr = placeHolderInfo.data();
    rtArg.placeHolderInfoNum = static_cast<uint32_t>(placeHolderInfo.size());
    // if input arg num is 0, dont need to report
    ReportRTSException(launchCache, PtrShift(rawArg, devAddrLen + currHostDataLen));

    size_t exceptionCacheLen = launchCache->GetExceptionArgNum() * Int64Btyes;
    void* dfxInfoCache = PtrShift(rawArg, devAddrLen + currHostDataLen + exceptionCacheLen);
    UpdateDFXInfoDumpAndTilingData(rtArg, launchCache, dfxInfoCache);

    OP_LOGI("%d", PrintRtArg(rtArg));
    auto& launchCfg = launchCache->GetKernelLaunchConfig();
    CHECK_RET_CODE(LaunchKernelFromCache(stream, rtArg, launchCfg), "Kernel launch from cache failed.");

    return ACLNN_SUCCESS;
}
} // namespace op::internal
