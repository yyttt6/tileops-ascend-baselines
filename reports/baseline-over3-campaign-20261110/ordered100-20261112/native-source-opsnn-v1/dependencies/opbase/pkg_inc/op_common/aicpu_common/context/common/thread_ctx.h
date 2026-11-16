/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_THREAD_CTX_H_
#define AICPU_CONTEXT_COMMON_THREAD_CTX_H_
#include "cpu_types.h"
#include "aicpu_context.h"

namespace aicpu {
class ThreadCtx {
public:
    explicit ThreadCtx(DeviceType device) : device_(device) {}

    virtual ~ThreadCtx() = default;

    virtual uint32_t SetThreadCtxInfo(CtxType type, const std::string& key, const std::string& value) const = 0;

    virtual uint32_t GetThreadCtxInfo(CtxType type, const std::string& key, std::string& value) const = 0;

    virtual uint32_t RemoveThreadCtxInfo(CtxType type, const std::string& key) const = 0;

private:
    ThreadCtx(const ThreadCtx&) = delete;
    ThreadCtx(ThreadCtx&&) = delete;
    ThreadCtx& operator=(const ThreadCtx&) = delete;
    ThreadCtx& operator=(ThreadCtx&&) = delete;

private:
    DeviceType device_; // device type, HOST/DEVICE
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_THREAD_CTX_H_
