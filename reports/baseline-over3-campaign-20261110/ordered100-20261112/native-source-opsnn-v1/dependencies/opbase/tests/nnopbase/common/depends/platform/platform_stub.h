/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PLATFORM_STUB_H_
#define PLATFORM_STUB_H_

#include <string>
#include <memory>

class PlatformInfoStub {
public:
    virtual ~PlatformInfoStub() = default;
    static PlatformInfoStub* GetInstance()
    {
        if (instance_ == nullptr) {
            instance_ = std::make_shared<PlatformInfoStub>();
        }
        return instance_.get();
    }

    void Reset()
    {
        shortSoCVersion_ = "Ascend910";
        soCVersion_ = "Ascend910B2";
    }

    void SetSoCVersion(std::string shortSoCVer, std::string soCVer)
    {
        shortSoCVersion_ = shortSoCVer;
        soCVersion_ = soCVer;
    }

    void GetSoCVersion(std::string& shortSoCVer, std::string& soCVer) const
    {
        shortSoCVer = shortSoCVersion_;
        soCVer = soCVersion_;
    }

private:
    thread_local static std::shared_ptr<PlatformInfoStub> instance_;
    std::string shortSoCVersion_{"Ascend910"};
    std::string soCVersion_{"Ascend910B2"};
};

#endif
