/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILER_STUB_H_
#define PROFILER_STUB_H_

#include <memory>
#include "profiling/aprof_pub.h"

class ProfilerStub {
public:
    virtual ~ProfilerStub() = default;
    static ProfilerStub* GetInstance()
    {
        if (fake_instance_ != nullptr) {
            return fake_instance_;
        }
        if (instance_ == nullptr) {
            instance_ = std::make_shared<ProfilerStub>();
        }
        return instance_.get();
    }

    virtual int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api) { return 0; }

    virtual int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

    virtual int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

    virtual int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName) { return 0; }

    void Install(ProfilerStub* instance) { fake_instance_ = instance; }

    void UnInstall() { fake_instance_ = nullptr; }

private:
    static std::shared_ptr<ProfilerStub> instance_;
    static ProfilerStub* fake_instance_;
};

#endif
