/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DUMP_STUB_H_
#define DUMP_STUB_H_

#include <memory>
#include "acl/acl_base.h"
#include "dump/adump_pub.h"
#include "dump/adump_api.h"
#include "profiling/aprof_pub.h"

namespace Adx {
class DumpStub {
public:
    virtual ~DumpStub() = default;
    static DumpStub* GetInstance()
    {
        if (fake_instance_ != nullptr) {
            return fake_instance_;
        }
        if (instance_ == nullptr) {
            instance_ = std::make_shared<DumpStub>();
        }
        return instance_.get();
    }

    virtual int32_t AdumpDumpTensorV2(const std::string& opType, const std::string& opName,
                                      const std::vector<Adx::TensorInfoV2>& tensors, aclrtStream stream)
    {
        return 0;
    }

    virtual int32_t AdumpAddExceptionOperatorInfoV2(const OperatorInfoV2& opInfo) { return 0; }

    virtual void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
    {
        atomicIndex = 1;
        return (void*)inputAddr_;
    }

    virtual void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
    {
        atomicIndex = 5525;
        return (void*)dfxAddr_;
    }

    void Install(DumpStub* instance) { fake_instance_ = instance; }

    void UnInstall() { fake_instance_ = nullptr; }

    virtual uint64_t AdumpGetDumpSwitch(DumpType type) { return 1; }

    virtual void AdumpPrintWorkSpace(const void* workSpaceAddr, const size_t dumpWorkSpaceSize, aclrtStream stream,
                                     const char* opType)
    {
        return;
    }

    virtual void AdumpPrintAndGetTimeStampInfo(const void* workSpaceAddr, const size_t dumpWorkSpaceSize,
                                               aclrtStream stream, const char* opType,
                                               std::vector<MsprofAicTimeStampInfo>& timeStampInfo)
    {
        return;
    }

    virtual bool AdumpIsDumpEnable(DumpType type) { return false; }

private:
    thread_local static std::shared_ptr<DumpStub> instance_;
    thread_local static DumpStub* fake_instance_;
    char inputAddr_[1000] = {0};
    char dfxAddr_[5000] = {0};
};
} // namespace Adx

#endif
