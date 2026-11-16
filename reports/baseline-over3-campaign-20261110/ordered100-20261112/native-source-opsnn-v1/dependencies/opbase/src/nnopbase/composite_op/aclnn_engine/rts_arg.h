/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_OPDEV_INTERNAL_RTS_LAUNCHER_H
#define OP_API_COMMON_INC_OPDEV_INTERNAL_RTS_LAUNCHER_H

#include <memory>
#include <vector>

#include "securec.h"
#include "acl/acl_rt.h"

#include "aclnn/acl_meta.h"
#include "kernel_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_cache.h"
#include "opdev/op_log.h"
#include "kernel_arg.h"
#include "op_ctx_def.h"

namespace op::internal {

constexpr size_t KERNEL_NAME_MAX_SIZE = 160;
constexpr int32_t LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2;

enum class LaunchKernelEngineType : uint8_t { NO_VECTOR_CORE = 0, VECTOR_CORE_ENGINE_AIC, VECTOR_CORE_ENGINE_AIV };

struct KernelLaunchConfig {
    aclrtBinHandle binHandle;
    aclrtFuncHandle funcHandle;
    uint64_t tilingKey;
    const char* kernelNameOfNoFatBin;
    uint32_t numBlocks;
    bool isFatBin;
    uint8_t schemMode;
    uint32_t dynUBufSize;
    uint32_t blockDimOffset;
    LaunchKernelEngineType engineType;
};

// rts args struct
using rtArgs_t = struct tagRtArgs {
    void* args;                               // args host mem addr
    aclrtPlaceHolderInfo* placeHolderInfoPtr; // nullptr means no host mem input
    uint32_t argsSize{0};                     // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset{0};             // tiling addr offset
    uint32_t tilingDataOffset{0};             // tiling data offset
    uint32_t placeHolderInfoNum{0};           // placeHolderInfo num
    uint8_t hasTiling{0};                     // if has tiling: 0 means no tiling
};

bool IsNeedOverflowStatusAddr();

class RtsApiFlag {
public:
    static RtsApiFlag& GetRtsApiFlag()
    {
        static RtsApiFlag flag;
        return flag;
    }

    RtsApiFlag() = default;

    void UseNewApi(bool flag) { useNewFlag_ = flag; }

    bool IfNewApi() const { return useNewFlag_; }

private:
    bool useNewFlag_{false};
};

class LaunchArgCache {
public:
    enum ArgType : uint32_t {
        FFTS_ADDR = 1,
        DEV_ADDR = 2,
        HOST_DATA = 3,
        TILING_DATA = 4,
        OVERFLOW_ADDR = 5,
        DEV_PTR_ADDR = 6
    };

    enum RtsApiType : uint16_t { RTS_OLD = 1, RTS_NEW = 2 };

    struct ArgInfo {
        ArgType type;     // ffts/device addr/host data/overflow/device ptr addr
        uint32_t dataLen; // only data on host has a length. dataLen of device data is zero
    };

    ArgInfo* GetArgInfo() { return PtrCastTo<ArgInfo>(argData_); }

    void SetArgNum(size_t num) { argNum_ = num; }

    size_t GetArgInfoNum() const { return argNum_; }

    void SetExceptionArgNum(size_t num) { exceptionArgNum_ = num; }

    size_t GetExceptionArgNum() const { return exceptionArgNum_; }

    void SetLaunchArgNum(size_t num) { launchArgNum_ = num; }

    size_t GetLaunchArgNum() const { return launchArgNum_; }

    void SetDFXInfoCacheElemCount(uint32_t count) { dfxInfoCacheElemCount_ = count; }

    uint32_t GetDFXInfoCacheElemCount() const { return dfxInfoCacheElemCount_; }

    void SetDFXInfoOffsetInTilingData(size_t offset) { dfxInfoOffsetInTilingData_ = offset; }

    size_t GetDFXInfoOffsetInTilingData() const { return dfxInfoOffsetInTilingData_; }

    void SetRtsApiType(RtsApiType type) { rtsType_ = type; }

    RtsApiType GetRtsApiType() const { return rtsType_; }

    bool SetRunParam(KernelLaunchConfig& launchCfg, std::string& kernelName)
    {
        launchCfg_ = launchCfg;
        OP_LOGD("Save launch config to cache, engine type: %d, isFatBin: %d, tilingKey: %lu, numBlocks: %u, "
                "scheduleMode: %u, blockDimOffset: %u, dynUBufSize: %u, funcHandle: %p, binHandle: %p",
                static_cast<int>(launchCfg_.engineType), launchCfg_.isFatBin, launchCfg_.tilingKey,
                launchCfg_.numBlocks, launchCfg_.schemMode, launchCfg_.blockDimOffset, launchCfg_.dynUBufSize,
                launchCfg_.funcHandle, launchCfg_.binHandle);
        if (!launchCfg_.isFatBin) {
            OP_CHECK(kernelName.length() < KERNEL_NAME_MAX_SIZE,
                     OP_LOGW("kernel name is bigger than %zu", KERNEL_NAME_MAX_SIZE), return false);
            auto rc = strncpy_s(kernelNameOfNoFatBin_, KERNEL_NAME_MAX_SIZE - 1, kernelName.c_str(),
                                kernelName.length());
            OP_CHECK(rc == EOK, OP_LOGW("Copy kernel name to cache failed"), return false);
            OP_LOGD("Kernel name of no fat bin is: %s", kernelNameOfNoFatBin_);
        }
        launchCfg_.kernelNameOfNoFatBin = kernelNameOfNoFatBin_;
        return true;
    }

    const char* GetKernelNameOfNoFatBin() const { return kernelNameOfNoFatBin_; }

    void* GetRawRtsArg() { return PtrShift(argData_, argNum_ * sizeof(ArgInfo)); }

    void* GetRawHostData() { return PtrShift(argData_, argNum_ * sizeof(ArgInfo) + argNum_ * sizeof(void*)); }

    void* GetLaunchHandle() const { return handle_; }

    uint32_t GetNumBlocks() const { return numBlocks_; }

    uint32_t GetDynUBufSize() const { return dynUBufSize_; }

    uint64_t GetTilingKey() const { return tilingKey_; }

    KernelLaunchConfig& GetKernelLaunchConfig()
    {
        launchCfg_.kernelNameOfNoFatBin = kernelNameOfNoFatBin_;
        return launchCfg_;
    }

    size_t GetRawArgSize()
    {
        size_t hostLen = 0;
        ArgInfo* argInfo = GetArgInfo();

        for (size_t i = 0; i < argNum_; i++) {
            if (argInfo[i].type == HOST_DATA || argInfo[i].type == TILING_DATA || argInfo[i].type == DEV_PTR_ADDR) {
                hostLen += argInfo[i].dataLen;
            }
        }
        return sizeof(void*) * argNum_ + hostLen;
    }

    void SetOpType(const char* opType) { strcpy_s(opType_, sizeof(opType_), opType); }

    const char* GetOpType() { return opType_; }

    void SetHostArgInfo(size_t size, size_t num)
    {
        hostArgSize_ = size;
        hostArgNum_ = num;
    }

    size_t GetHostArgSize() const { return hostArgSize_; }

    size_t GetHostArgNum() const { return hostArgNum_; }

    static aclnnStatus UpdateFunctionHandle(KernelLaunchConfig& launchCfg);
    static aclnnStatus LaunchKernelFromCache(aclrtStream stream, rtArgs_t& rtArg, KernelLaunchConfig& launchCfg);
    static aclnnStatus RunFromCache(aclrtStream stream, void* cache);

private:
    char opType_[16] = {0};
    char kernelNameOfNoFatBin_[KERNEL_NAME_MAX_SIZE] = {0};
    void* handle_{nullptr};
    KernelLaunchConfig launchCfg_;

    RtsApiType rtsType_{0};
    uint32_t numBlocks_{0};
    uint64_t tilingKey_{0};
    uint32_t dynUBufSize_{0};

    size_t hostArgSize_{0};
    size_t hostArgNum_{0};

    size_t argNum_{0};
    size_t exceptionArgNum_{0};
    size_t launchArgNum_{0};
    uint32_t dfxInfoCacheElemCount_{0};
    size_t dfxInfoOffsetInTilingData_{0};
    uint8_t argData_[0]; // ArgInfo array + dev_addr + host data
};

// There's how launch args are composed:
// ffts_addr, input_addrs, output_addrs, outshape_addrs, workspace_addrs, tiling_addr, overflow_addr, host_data...
// Note: constructor may throw std::bad_alloc if buffer expansion fails
class RtsArg {
public:
    explicit RtsArg(bool hasFftsAddr, const LaunchArgInfo& argInfo, ExpandableRtsArgBuffer* rtsArgBuffer);
    ~RtsArg();
    aclnnStatus FillArgs(bool assertFlag = false);
    LaunchArgCache* DumpToCache();

    void ReportExceptionDumpInfo() const;
    aclnnStatus LaunchKernel(aclrtStream stream, const KernelLaunchConfig& launchCfg);

    const rtArgs_t& GetRtsArg() const { return rtArg_; }

    std::vector<int32_t> GetTensorOffset() const { return tensorOffset_; }
    static constexpr size_t HOST_VALUE_ALIGNMENT = 32;
    static constexpr uint32_t DEV_PTR_DIM_SHIFT_BIT = 32;

private:
    aclnnStatus AppendFftsAddr();
    aclnnStatus FinalizeArg();
    void AppendExceptionDumpAddr(bool assertFlag = false);
    void AppendArg(void* arg)
    {
        *hostAddr_ = arg;
        tensorOffset_.push_back(hostAddrOffset_ / PTR_SIZE);
        hostAddr_++;
        hostAddrOffset_ += PTR_SIZE;
    }

    aclnnStatus AppendHostArg(void* hostData, size_t hostDataSize);
    aclnnStatus AppendDevicePtrArg(const aclTensorList* tensors, size_t dataSize);
    void AppendOverflowStatusAddr()
    {
        void* overflowAddr = nullptr;
        static bool needOverflowAddr = IsNeedOverflowStatusAddr();
        if (needOverflowAddr) {
            aclError rc = aclrtCtxGetFloatOverflowAddr(&overflowAddr);
            OP_CHECK(rc == ACL_RT_NO_ERROR, OP_LOGW("aclrtCtxGetFloatOverflowAddr failed. %d", rc), return);
        }
        *hostAddr_ = overflowAddr;
        hostAddr_++;
        hostAddrOffset_ += PTR_SIZE;
    }

    // 刷新 rtArg_.args 和 hostAddr_（扩容后 baseAddr_ 改变时调用）
    void RefreshRtArgsAddr();

    void AddExceptionDumpDataToCache(const LaunchArgInfo& argInfo, OpExecCache* cache,
                                     LaunchArgCache* launchCache) const;
    void AddDFXInfoDumpDataToCache(const LaunchArgInfo& argInfo, OpExecCache* cache, LaunchArgCache* launchCache) const;

    bool hasFftsAddr_{false};
    const LaunchArgInfo& argInfo_;
    rtArgs_t rtArg_;
    size_t argNum_;

    void** hostAddr_{nullptr};
    std::vector<int32_t> tensorOffset_;

    size_t hostAddrOffset_{0}; // hostAddr_ 相对于 rtArg_.args 的偏移量（字节）
    void* exceptionDumpAddr_{nullptr};
    uint32_t exceptionDumpIndex_{0};

    ExpandableRtsArgBuffer* rtsArgBuffer_{nullptr};

    static constexpr size_t MAX_HOST_INFO_NUM = 16;
    static constexpr size_t PTR_SIZE = 8;

public:
    thread_local static std::vector<aclrtPlaceHolderInfo> placeHolderInfo_;
};

int PrintRtArg(const rtArgs_t& rtArg);
int PrintExceptionDumpInfo(void* dump, size_t num);
int PrintAICErrorDFXInfo(const void* dfxInfoAddr, const size_t argNum, const size_t dataSize);
void PrintHostDataSize(const rtArgs_t& rtArg);
void PrintTilingData(const rtArgs_t& rtArg);
void AddArgInfoToCache(OpExecCache* cache, LaunchArgCache::ArgInfo* argInfo, const LaunchArgInfo& launchArgInfo,
                       bool hasFftsAddr, const ExpandableRtsArgBuffer* rtsArgBuffer);
void ReportRTSException(const LaunchArgCache* launchCache, void* cacheException);
} // namespace op::internal

#endif