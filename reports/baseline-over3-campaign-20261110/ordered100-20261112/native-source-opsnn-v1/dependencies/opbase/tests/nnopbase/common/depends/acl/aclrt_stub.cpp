/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "aclrt_stub.h"
#include "securec.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

thread_local std::shared_ptr<AclrtStub> AclrtStub::aclrtInstance_;
thread_local AclrtStub* AclrtStub::fakeAclrtInstance_;

#ifndef EXTERN_C
#ifdef OP_API_COMMON_UT_NEED_EXTERN_C
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
#endif
char aclSocVersion[100];
EXTERN_C
aclError aclrtSetDevice(int32_t deviceId) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtGetCurrentContext(aclrtContext* context) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtSetCurrentContext(aclrtContext context) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtCreateStream(aclrtStream* stream) { return ACL_SUCCESS; }

const char* aclrtGetSocName()
{
    std::string str = "Ascend910A";
    const char* value = std::getenv("ASCEND_C");
    if (value != nullptr) {
        const std::string valStr = std::string(value);
        if (valStr == "1") {
            str = "Ascend910B1";
        } else if (valStr == "2") {
            str = "Ascend950DT_9591";
        } else if (valStr == "3") {
            str = "Ascend310P1";
        } else if (valStr == "999") {
            return nullptr;
        }
    }
    strncpy(aclSocVersion, str.c_str(), 100);
    return aclSocVersion;
}

EXTERN_C
aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return ACL_SUCCESS;
}
EXTERN_C
aclError aclrtSynchronizeStream(aclrtStream stream) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    aclrtMemcpyKind kindInfo[4] = {
        ACL_MEMCPY_HOST_TO_HOST,     // ACL_MEMCPY_HOST_TO_HOST==RT_MEMCPY_HOST_TO_HOST,
        ACL_MEMCPY_HOST_TO_DEVICE,   // ACL_MEMCPY_HOST_TO_DEVICE=RT_MEMCPY_HOST_TO_DEVICE,
        ACL_MEMCPY_DEVICE_TO_HOST,   // ACL_MEMCPY_DEVICE_TO_HOST=RT_MEMCPY_DEVICE_TO_HOST,
        ACL_MEMCPY_DEVICE_TO_DEVICE, // ACL_MEMCPY_DEVICE_TO_DEVICE=RT_MEMCPY_DEVICE_TO_DEVICE
    };
    if (kind != ACL_MEMCPY_DEVICE_TO_HOST) {
        memcpy(dst, src, count);
    } else if (destMax == 75 * 1024 * 1024 && count == 75 * 1024 * 1024) {
        memcpy(dst, src, count);
    } else if (std::getenv("MEMCPY_ENV") != nullptr) {
        memcpy(dst, src, count);
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtFree(void* devPtr)
{
    delete[] (uint8_t*)devPtr;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind,
                          aclrtStream stream)
{
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetThreadLastTaskId(uint32_t* taskId) { return ACL_SUCCESS; }
EXTERN_C
aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId) { return ACL_SUCCESS; }
EXTERN_C
aclError aclrtGetDevice(int32_t* device)
{
    *device = 0;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetHardwareSyncAddr(void** addr) { return ACL_SUCCESS; }

int64_t g_deterministic = 0;
int64_t g_debugKernel = 0;
EXTERN_C
aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    if (opt == ACL_OPT_DETERMINISTIC) {
        g_deterministic = value;
    } else if (opt == ACL_OPT_ENABLE_DEBUG_KERNEL) {
        g_debugKernel = value;
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t* value)
{
    if (opt == ACL_OPT_DETERMINISTIC) {
        *value = g_deterministic;
    } else if (opt == ACL_OPT_ENABLE_DEBUG_KERNEL) {
        *value = g_debugKernel;
    }
    return ACL_SUCCESS;
}

aclrtFloatOverflowMode g_floatOverflowMode = ACL_RT_OVERFLOW_MODE_SATURATION;
EXTERN_C
aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode* floatOverflowMode)
{
    *floatOverflowMode = g_floatOverflowMode;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode)
{
    g_floatOverflowMode = mode;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtMallocWithCfg(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                       aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                       aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    return AclrtStub::GetInstance()->aclrtLaunchKernelWithHostArgs(funcHandle, blockDim, stream, cfg, hostArgs,
                                                                   argsSize, placeHolderArray, placeHolderNum);
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
{
    return AclrtStub::GetInstance()->aclrtBinaryGetFunctionByEntry(binHandle, funcEntry, funcHandle);
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    return AclrtStub::GetInstance()->aclrtBinaryGetFunction(binHandle, kernelName, funcHandle);
}

aclError aclrtBinaryLoadFromData(const void* data, size_t length, const aclrtBinaryLoadOptions* options,
                                 aclrtBinHandle* binHandle)
{
    return AclrtStub::GetInstance()->aclrtBinaryLoadFromData(data, length, options, binHandle);
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    return AclrtStub::GetInstance()->aclrtBinaryLoadFromFile(binPath, options, binHandle);
}

aclError aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char* funcName, const char* kernelName,
                              aclrtFuncHandle* funcHandle)
{
    return AclrtStub::GetInstance()->aclrtRegisterCpuFunc(handle, funcName, kernelName, funcHandle);
}

EXTERN_C
aclError aclrtNpuGetFloatOverFlowStatus(void* outputAddr, uint64_t outputSize, uint32_t checkMode, aclrtStream stream)
{
    return ACL_SUCCESS;
}
EXTERN_C
aclError aclrtNpuClearFloatOverFlowStatus(uint32_t checkMode, aclrtStream stream) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtCtxGetFloatOverflowAddr(void** overflowAddr)
{
    *overflowAddr = (void*)0x005;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtRandomNumAsync(const aclrtRandomNumTaskInfo* taskInfo, const aclrtStream stream, void* reserve)
{
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtGetOpExecuteTimeout(uint32_t* const timeoutMs) { return ACL_SUCCESS; }

EXTERN_C
aclError aclrtBinarySetExceptionCallback(aclrtBinHandle binHandle, aclrtOpExceptionCallback callback, void* userData)
{
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
    *value = 8;
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return AclrtStub::GetInstance()->aclrtGetStreamAttribute(stream, stmAttrType, value);
}

EXTERN_C
aclError aclrtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize)
{
    return AclrtStub::GetInstance()->aclrtCacheLastTaskOpInfo(infoPtr, infoSize);
}

int64_t gDeterministicNew = 0;
int64_t gStrongConsistency = 0;

EXTERN_C
aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    if (opt == ACL_OPT_DETERMINISTIC) {
        gDeterministicNew = value;
    } else if (opt == ACL_OPT_STRONG_CONSISTENCY) {
        gStrongConsistency = value;
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t* value)
{
    if (opt == ACL_OPT_DETERMINISTIC) {
        *value = gDeterministicNew;
    } else if (opt == ACL_OPT_STRONG_CONSISTENCY) {
        *value = gStrongConsistency;
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* captureMdl)
{
    return AclrtStub::GetInstance()->aclmdlRICaptureGetInfo(stream, status, captureMdl);
}

EXTERN_C
aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    *stream = (void*)(new uint8_t[1]);
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t flag)
{
    *event = (void*)(new uint8_t[1]);
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtDestroyEvent(aclrtEvent event)
{
    if (event != nullptr) {
        delete[] static_cast<uint8_t*>(event);
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtCtxGetCurrentDefaultStream(aclrtStream* stream)
{
    *stream = (void*)(new uint8_t[1]);
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtDestroyStream(aclrtStream stream)
{
    if (stream != nullptr) {
        delete[] static_cast<uint8_t*>(stream);
    }
    return ACL_SUCCESS;
}

EXTERN_C
aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    if (value != nullptr) {
        *value = 24U;
    }
    return ACL_SUCCESS;
}
