/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_stub.h"
#include "securec.h"

#ifndef EXTERN_C
#ifdef OP_API_COMMON_UT_NEED_EXTERN_C
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
#endif

thread_local std::shared_ptr<RuntimeStub> RuntimeStub::instance_;
thread_local RuntimeStub* RuntimeStub::fake_instance_;

EXTERN_C
rtError_t rtGetSocVersion(char* version, const uint32_t maxLen)
{
    std::string str = "Ascend910A";
    char* value = std::getenv("ASCEND_C");
    if (value != nullptr) {
        if (strcmp(value, "1") == 0) {
            str = "Ascend910B1";
        } else if (strcmp(value, "2") == 0) {
            str = "Ascend950";
        } else if (strcmp(value, "3") == 0) {
            str = "Ascend310P1";
        }
    }
    strncpy(version, str.c_str(), maxLen);
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || strcmp(label, "version") != 0 || strcmp(key, "NpuArch") != 0) {
        return 1;
    }
    strncpy(val, arch, maxLen);
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtRegisterAllKernel(const rtDevBinary_t* bin, void** handle) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtDevBinaryUnRegister(void* handle) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t* argsInfo,
                                     rtSmDesc_t* smDesc, rtStream_t stm, const rtTaskCfgInfo_t* cfgInfo)
{
    return RuntimeStub::GetInstance()->rtKernelLaunchWithHandleV2(hdl, tilingKey, blockDim, argsInfo, smDesc, stm,
                                                                  cfgInfo);
}

EXTERN_C
rtError_t rtsGetResInCurrentThread(const rtDevResLimitType_t type, uint32_t* const value)
{
    return RuntimeStub::GetInstance()->rtsGetResInCurrentThread(type, value);
}

EXTERN_C
rtError_t rtKernelLaunchWithFlagV2(const void* stubFunc, uint32_t blockDim, rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc,
                                   rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtLaunchKernelByFuncHandleV2(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                       rtStream_t stm, const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtGetDevice(int32_t* device)
{
    *device = 0;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtCtxGetDevice(int32_t* device)
{
    *device = 0;
    if (std::getenv("ENABLE_1982")) {
        *device = 1;
    }
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t* freeSize, size_t* totalSize) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtDevBinaryRegister(const rtDevBinary_t* bin, void** hdl) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtFunctionRegister(void* binHandle, const void* stubFunc, const char_t* stubName, const void* kernelInfoExt,
                             uint32_t funcMode)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtGetFunctionByName(const char_t* stubName, void** stubFunc) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtGetDeviceInfo(uint32_t device_id, int32_t module_type, int32_t info_type, int64_t* val)
{
    *val = 8;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t* aiCoreMemorySize) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char* opName, uint32_t blockDim,
                                        const rtAicpuArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stream,
                                        uint32_t flags)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtStreamSynchronize(rtStream_t stream) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout) { return RT_ERROR_NONE; }
int64_t gDeterministic = 0;
int64_t gDebugKernel = 0;
rtFloatOverflowMode_t gOverflow = RT_OVERFLOW_MODE_SATURATION;

rtError_t rtCtxSetSysParamOpt(const rtSysParamOpt configOpt, const int64_t configVal)
{
    if (configOpt == SYS_OPT_DETERMINISTIC) {
        gDeterministic = configVal;
    } else if (configOpt == SYS_OPT_ENABLE_DEBUG_KERNEL) {
        gDebugKernel = configVal;
    }
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetSysParamOpt(const rtSysParamOpt configOpt, int64_t* const configVal)
{
    if (configOpt == SYS_OPT_DETERMINISTIC) {
        *configVal = gDeterministic;
    } else if (configOpt == SYS_OPT_ENABLE_DEBUG_KERNEL) {
        *configVal = gDebugKernel;
    }
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t* floatOverflowMode)
{
    *floatOverflowMode = gOverflow;
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetOverflowAddr(void** overflowAddr)
{
    *overflowAddr = (void*)0x005;
    return RT_ERROR_NONE;
}

rtError_t rtSetDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode)
{
    gOverflow = floatOverflowMode;
    return RT_ERROR_NONE;
}

rtError_t rtMalloc(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtFree(void* devptr)
{
    delete[] (uint8_t*)devptr;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtMemcpyAsync(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
{
    return RuntimeStub::GetInstance()->rtMemcpy(dst, destMax, src, count, kind);
}

EXTERN_C
rtError_t rtGeneralCtrl(uintptr_t* ctrl, uint32_t num, uint32_t type)
{
    return RuntimeStub::GetInstance()->rtGeneralCtrl(ctrl, num, type);
}

EXTERN_C
rtError_t rtCalcLaunchArgsSize(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, size_t* launchArgsSize)
{
    *launchArgsSize = argsSize + hostInfoTotalSize;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, void* argsData,
                             rtLaunchArgsHandle* argsHandle)
{
    static size_t hdlData = 0;
    static rtLaunchArgsHandle hdl = static_cast<void*>(&hdlData);
    *argsHandle = hdl;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtAppendLaunchAddrInfo(rtLaunchArgsHandle argsHandle, void* addrInfo) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtAppendLaunchHostInfo(rtLaunchArgsHandle argsHandle, size_t hostInfoSize, void** hostInfo)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtBinaryLoad(const rtDevBinary_t* bin, rtBinHandle* binHandle) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtBinaryGetFunction(const rtBinHandle binHandle, const uint64_t tilingKey, rtFuncHandle* funcHandle)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                     rtStream_t stm)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t* const sizeInfo) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtStreamDestroy(rtStream_t stream)
{
    if (stream != nullptr) {
        delete[] static_cast<uint8_t*>(stream);
    }
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtsNpuClearFloatOverFlowDebugStatus(uint32_t checkMode, rtStream_t stm) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtsFuncGetByEntry(const rtBinHandle binHandle, const uint64_t funcEntry, rtFuncHandle* funcHandle)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtsNpuGetFloatOverFlowDebugStatus(void* outputAddrPtr, uint64_t outputSize, uint32_t checkMode,
                                            rtStream_t stm)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtsBinaryLoadFromData(const void* const data, const uint64_t length,
                                const rtLoadBinaryConfig_t* const optionalCfg, rtBinHandle* handle)
{
    (void)data;
    (void)length;
    (void)optionalCfg;
    (void)handle;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtEventCreateWithFlag(rtEvent_t* event, uint32_t flag)
{
    *event = (void*)(new uint8_t[1]);
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtEventCreateExWithFlag(rtEvent_t* event, uint32_t flag) { return rtEventCreateWithFlag(event, flag); }

EXTERN_C
rtError_t rtEventDestroy(rtEvent_t event)
{
    if (event != nullptr) {
        delete[] static_cast<uint8_t*>(event);
    }
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtEventReset(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtRegStreamStateCallback(const char_t* regName, const rtStreamStateCallback callback)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtCtxGetCurrentDefaultStream(rtStream_t* stm) { return RT_ERROR_NONE; }

static uint64_t floatDebugStatus = 0;
EXTERN_C
rtError_t rtNpuGetFloatDebugStatus(void* outputAddrPtr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm)
{
    uint64_t* status = static_cast<uint64_t*>(outputAddrPtr);
    floatDebugStatus = 1;
    *status = floatDebugStatus;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtNpuClearFloatDebugStatus(uint32_t checkMode, rtStream_t stm)
{
    floatDebugStatus = 0;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtCtxGetCurrent(rtContext_t* ctx)
{
    int x = 1;
    *ctx = (void*)x;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtVectorCoreKernelLaunchWithHandle(void* hdl, const uint64_t tilingKey, uint32_t blockDim,
                                             rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stm,
                                             const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtVectorCoreKernelLaunch(const void* stubFunc, uint32_t blockDim, rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc,
                                   rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtStreamGetMode(rtStream_t const stm, uint64_t* const stmMode) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stream) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtNotifyRecord(rtNotify_t notify, rtStream_t stream) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtGetOpExecuteTimeOut(uint32_t* const timeout) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus* const status, rtModel_t* captureMdl)
{
    *status = RT_STREAM_CAPTURE_STATUS_ACTIVE;
    return RT_ERROR_NONE;
}

EXTERN_C
rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) { return RT_ERROR_NONE; }

EXTERN_C
rtError_t rtFusionLaunch(void* const fusionInfo, rtStream_t const stm, rtFusionArgsEx_t* argsInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtsBinaryLoadFromFile(const char_t* const binPath, const rtLoadBinaryConfig_t* const optionalCfg,
                                rtBinHandle* handle)
{
    return RT_ERROR_NONE;
}

rtError_t rtsLaunchCpuKernel(const rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                             const rtKernelLaunchCfg_t* cfg, rtCpuKernelArgs_t* argsInfo)
{
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsCtxGetFloatOverflowAddr(void** overflowAddr)
{
    *overflowAddr = (void*)0x005;
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsEventWait(rtStream_t stream, rtEvent_t evt, uint32_t timeout) { return RT_ERROR_NONE; }

extern "C" rtError_t rtsEventRecord(rtEvent_t evt, rtStream_t stm) { return RT_ERROR_NONE; }

extern "C" rtError_t rtsGetSocVersion(char_t* ver, const uint32_t maxLen)
{
    std::string str = "Ascend910A";
    if (std::getenv("ASCEND_C") != nullptr) {
        str = "Ascend910B1";
    }
    strncpy(ver, str.c_str(), maxLen);
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsCtxGetSysParamOpt(const rtSysParamOpt configOpt, int64_t* const configVal)
{
    if (configOpt == SYS_OPT_DETERMINISTIC) {
        *configVal = gDeterministic;
    } else if (configOpt == SYS_OPT_ENABLE_DEBUG_KERNEL) {
        *configVal = gDebugKernel;
    }
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsGetDevice(int32_t* devId)
{
    *devId = 0;
    if (std::getenv("ENABLE_1982")) {
        *devId = 1;
    }
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsGetInterCoreSyncAddr(uint64_t* addr, uint32_t* len) { return RT_ERROR_NONE; }

extern "C" rtError_t rtsFree(void* devptr)
{
    delete[] (uint8_t*)devptr;
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsMalloc(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsStreamSynchronize(rtStream_t stream, int32_t timeout) { return RT_ERROR_NONE; }

extern "C" rtError_t rtsNpuClearFloatOverFlowStatus(uint32_t checkMode, rtStream_t stm) { return RT_ERROR_NONE; }

extern "C" rtError_t rtsNpuGetFloatOverFlowStatus(void* outputAddrPtr, uint64_t outputSize, uint32_t checkMode,
                                                  rtStream_t stm)
{
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtsLaunchRandomNumTask(const rtRandomNumTaskInfo_t* taskInfo, const rtStream_t stm, void* reserve)
{
    return RT_ERROR_NONE;
}
