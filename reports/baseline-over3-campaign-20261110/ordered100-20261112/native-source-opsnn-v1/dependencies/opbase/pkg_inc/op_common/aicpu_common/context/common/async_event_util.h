/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_
#define AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_

#include <functional>
#include "aicpu_context.h"

namespace aicpu {
using NotifyWaitFunc = void (*)(void* notify_param, const uint32_t param_len);
using RegEventCbFunc = bool (*)(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void*)>& cb);
using RegEventCbWithTimesFunc = bool (*)(const uint32_t event_id, const uint32_t sub_event_id,
                                         const std::function<void(void*)>& cb, const int32_t times);
using UnregEventCbFunc = void (*)(const uint32_t event_id, const uint32_t sub_event_id);

template <typename TI, typename TO>
inline auto PtrToFunc(TI* const ptr) -> TO
{
    return reinterpret_cast<TO>(ptr);
}

class AsyncEventUtil {
public:
    static AsyncEventUtil& GetInstance();

    void NotifyWait(void* notify_param, const uint32_t param_len) const;

    bool RegEventCb(const uint32_t event_id, const uint32_t sub_event_id, const std::function<void(void*)>& cb);

    bool RegEventCb(const uint32_t event_id, const uint32_t sub_event_id, const std::function<void(void*)>& cb,
                    const int32_t times);

    void UnregEventCb(const uint32_t event_id, const uint32_t sub_event_id);

    bool RegOpEventCb(const uint32_t event_id, const uint32_t sub_event_id, const std::function<void(void*)>& cb) const;

    void UnregOpEventCb(const uint32_t event_id, const uint32_t sub_event_id) const;

private:
    AsyncEventUtil();
    ~AsyncEventUtil();
    void InitEventUtil();
    void* sharder_;
    NotifyWaitFunc notify_wait_func_;
    RegEventCbFunc reg_event_cb_func_;
    RegEventCbWithTimesFunc reg_event_cb_with_times_func_;
    UnregEventCbFunc unreg_event_cb_func_;
    RegEventCbFunc reg_op_event_cb_func_;
    UnregEventCbFunc unreg_op_event_cb_func_;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_ASYNC_EVENT_H_