/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump_stub.h"
#include "opdev/fp16_t.h"

namespace Adx {
thread_local std::shared_ptr<DumpStub> DumpStub::instance_;
thread_local DumpStub* DumpStub::fake_instance_;

int32_t AdumpDumpTensorV2(const std::string& opType, const std::string& opName,
                          const std::vector<TensorInfoV2>& tensors, aclrtStream stream)
{
    return DumpStub::GetInstance()->AdumpDumpTensorV2(opType, opName, tensors, stream);
}

int32_t AdumpAddExceptionOperatorInfoV2(const OperatorInfoV2& opInfo)
{
    return DumpStub::GetInstance()->AdumpAddExceptionOperatorInfoV2(opInfo);
}

void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
{
    return DumpStub::GetInstance()->AdumpGetSizeInfoAddr(space, atomicIndex);
}
void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
{
    return DumpStub::GetInstance()->AdumpGetDFXInfoAddrForDynamic(space, atomicIndex);
}
uint64_t AdumpGetDumpSwitch(DumpType type) { return DumpStub::GetInstance()->AdumpGetDumpSwitch(type); }
void AdumpPrintWorkSpace(const void* workSpaceAddr, const size_t dumpWorkSpaceSize, aclrtStream stream,
                         const char* opType)
{
    auto dumpStub = DumpStub::GetInstance();
    dumpStub->AdumpPrintWorkSpace(workSpaceAddr, dumpWorkSpaceSize, stream, opType);
}

bool AdumpIsDumpEnable(DumpType type) { return DumpStub::GetInstance()->AdumpIsDumpEnable(type); }

void AdumpPrintAndGetTimeStampInfo(const void* workSpaceAddr, const size_t dumpWorkSpaceSize, aclrtStream stream,
                                   const char* opType, std::vector<MsprofAicTimeStampInfo>& timeStampInfo)
{
    auto dumpStub = DumpStub::GetInstance();
    dumpStub->AdumpPrintAndGetTimeStampInfo(workSpaceAddr, dumpWorkSpaceSize, stream, opType, timeStampInfo);
}
} // namespace Adx
