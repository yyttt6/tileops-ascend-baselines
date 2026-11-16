/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MMPA_STUB_H_
#define MMPA_STUB_H_

#include <memory>
#include "acl/acl_base.h"
#include "hccl/hccl_types.h"

namespace Adx {

class MmpaStub {
public:
    virtual ~MmpaStub() = default;
    static MmpaStub* GetInstance()
    {
        if (fake_instance_ != nullptr) {
            return fake_instance_;
        }
        if (instance_ == nullptr) {
            instance_ = std::make_shared<MmpaStub>();
        }
        return instance_.get();
    }

    void Install(MmpaStub* instance) { fake_instance_ = instance; }

    void UnInstall() { fake_instance_ = nullptr; }

    virtual void* mmDlsym(void* handle, const char* funcName) { return nullptr; }

    virtual void* mmDlopen(const char* fileName, int32_t mode)
    {
        static int64_t tmp = 0;
        return &tmp;
    }

private:
    static std::shared_ptr<MmpaStub> instance_;
    static MmpaStub* fake_instance_;
};

} // namespace Adx

#endif
