/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLRT_STUB_H_
#define ACLRT_STUB_H_

#include <memory>
#include <string>
#include <algorithm>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"

#include "opdev/op_log.h"

class AclrtStub {
public:
    virtual ~AclrtStub() = default;
    static AclrtStub* GetInstance()
    {
        if (fakeAclrtInstance_ != nullptr) {
            return fakeAclrtInstance_;
        }
        if (aclrtInstance_ == nullptr) {
            aclrtInstance_ = std::make_shared<AclrtStub>();
        }
        return aclrtInstance_.get();
    }

    virtual aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry,
                                                   aclrtFuncHandle* funcHandle)
    {
        *funcHandle = (void*)0x12341234;
        return ACL_SUCCESS;
    }

    virtual aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName,
                                            aclrtFuncHandle* funcHandle)
    {
        CHECK_COND(kernelName != nullptr, ACL_ERROR_INVALID_PARAM, "Parameter kernelName is nullptr.");
        std::string name(kernelName);
        CHECK_COND(std::count(name.begin(), name.end(), '_') > 0, ACL_ERROR_INVALID_PARAM,
                   "Parameter kernelName is invalid, name: %s", kernelName);
        *funcHandle = (void*)0x43214321;
        return ACL_SUCCESS;
    }

    virtual aclError aclrtBinaryLoadFromData(const void* data, size_t length, const aclrtBinaryLoadOptions* options,
                                             aclrtBinHandle* binHandle)
    {
        return ACL_SUCCESS;
    }

    virtual aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options,
                                             aclrtBinHandle* binHandle)
    {
        return ACL_SUCCESS;
    }

    virtual aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                                   aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                                   aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        return ACL_SUCCESS;
    }

    virtual aclError aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char* funcName, const char* kernelName,
                                          aclrtFuncHandle* funcHandle)
    {
        return ACL_SUCCESS;
    }

    void Install(AclrtStub* instance) { fakeAclrtInstance_ = instance; }

    void UnInstall() { fakeAclrtInstance_ = nullptr; }

    virtual aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType,
                                             aclrtStreamAttrValue* value)
    {
        value->cacheOpInfoSwitch = 1;
        return ACL_SUCCESS;
    }

    virtual aclError aclrtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize) { return ACL_SUCCESS; }

    virtual aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* captureMdl)
    {
        *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        return ACL_SUCCESS;
    }

private:
    thread_local static std::shared_ptr<AclrtStub> aclrtInstance_;
    thread_local static AclrtStub* fakeAclrtInstance_;
};

#endif
