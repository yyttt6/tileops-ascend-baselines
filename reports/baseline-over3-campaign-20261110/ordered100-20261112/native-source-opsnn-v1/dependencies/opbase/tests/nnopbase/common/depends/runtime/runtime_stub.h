/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_STUB_H_
#define RUNTIME_STUB_H_

#include <cstring>
#include <string.h>
#include <string>
#include <memory>
#include "runtime/rt.h"
#include "runtime/rts/rts_kernel.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_stars.h"

class RuntimeStub {
    virtual ~RuntimeStub() = default;
    static RuntimeStub* GetInstance()
    {
        if (fake_instance_ != nullptr) {
            return fake_instance_;
        }
        if (instance_ == nullptr) {
            instance_ = std::make_shared<RuntimeStub>();
        }
        return instance_.get();
    }

    virtual rtError_t rtGeneralCtrl(uintptr_t* ctrl, uint32_t num, uint32_t type) { return RT_ERROR_NONE; }

    virtual rtError_t rtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
    {
        if (kind != RT_MEMCPY_DEVICE_TO_HOST) {
            memcpy(dst, src, count);
        } else if (destMax == 75 * 1024 * 1024 && count == 75 * 1024 * 1024) {
            memcpy(dst, src, count);
        } else if (std::getenv("MEMCPY_ENV") != nullptr) {
            memcpy(dst, src, count);
        }
        return RT_ERROR_NONE;
    }

    virtual rtError_t rtKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t blockDim,
                                                 rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stm,
                                                 const rtTaskCfgInfo_t* cfgInfo)
    {
        return RT_ERROR_NONE;
    }

    virtual rtError_t rtsGetResInCurrentThread(const rtDevResLimitType_t type, uint32_t* const value)
    {
        *value = 24;
        return RT_ERROR_NONE;
    }

    void Install(RuntimeStub* instance) { fake_instance_ = instance; }

    void UnInstall() { fake_instance_ = nullptr; }

private:
    thread_local static std::shared_ptr<RuntimeStub> instance_;
    thread_local static RuntimeStub* fake_instance_;
};

#endif
