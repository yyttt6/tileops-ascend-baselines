/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "profiler_stub.h"

std::shared_ptr<ProfilerStub> ProfilerStub::instance_;
ProfilerStub* ProfilerStub::fake_instance_;

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
{
    return ProfilerStub::GetInstance()->MsprofReportApi(agingFlag, api);
}

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    return ProfilerStub::GetInstance()->MsprofReportAdditionalInfo(agingFlag, data, length);
}

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    return ProfilerStub::GetInstance()->MsprofReportCompactInfo(agingFlag, data, length);
}

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName)
{
    return ProfilerStub::GetInstance()->MsprofRegTypeInfo(level, typeId, typeName);
}
