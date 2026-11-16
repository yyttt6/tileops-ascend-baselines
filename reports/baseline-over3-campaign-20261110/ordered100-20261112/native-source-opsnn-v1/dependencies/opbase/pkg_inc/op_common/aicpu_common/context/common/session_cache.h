/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_SESSION_CACHE_H
#define AICPU_CONTEXT_COMMON_SESSION_CACHE_H

#include <map>
#include <mutex>
#include <list>
#include "kernel_cache.h"

namespace {
constexpr uint16_t kSessionCacheNum = 256;
}
namespace aicpu {
template <class C>
class SessionCache {
public:
    static SessionCache<C>& Instance()
    {
        static SessionCache<C> instance;
        return instance;
    }

    /*
     * run and cache kernel.
     * @param param: kernel context
     * @param session_id: sesson id
     * @param stream_id: stream id
     * @param sess_flag: whether it's a session scene, true use session id, false
     * @param blkdim_info: Op's blkdim_info
     * use stream id
     * @return int32_t: 0 indicates success, while the others fail
     */
    template <class T>
    int32_t RunCpuKernelWithBlock(void* param, uint64_t session_id, uint64_t stream_id, bool sess_flag,
                                  struct BlkDimInfo* blkdim_info)
    {
        std::shared_ptr<KernelCache<C>> kernel = nullptr;
        if (sess_flag) {
            KERNEL_LOG_DEBUG("SessionCache KernelCache from session, id[%lu].", session_id);
            std::unique_lock<std::mutex> lock(session_mutex_);
            int32_t ret = GetOrCreateKernelCache<T>(session_kernel_cache_, session_id, sess_flag, kernel);
            if (ret != 0) {
                return ret;
            }
        } else {
            KERNEL_LOG_DEBUG("SessionCache KernelCache from stream, id[%lu].", stream_id);
            std::unique_lock<std::mutex> lock(stream_mutex_);
            int32_t ret = GetOrCreateKernelCache<T>(stream_kernel_cache_, stream_id, sess_flag, kernel);
            if (ret != 0) {
                return ret;
            }
        }
        return kernel->RunCpuKernelWithBlock(param, blkdim_info);
    }

    /*
     * run and cache kernel.
     * @param param: kernel context
     * @param session_id: sesson id
     * @param stream_id: stream id
     * @param sess_flag: whether it's a session scene, true use session id, false
     * use stream id
     * @return int32_t: 0 indicates success, while the others fail
     */
    template <class T>
    int32_t RunKernel(void* param, uint64_t session_id, uint64_t stream_id, bool sess_flag)
    {
        std::shared_ptr<KernelCache<C>> kernel = nullptr;
        if (sess_flag) {
            KERNEL_LOG_DEBUG("SessionCache KernelCache from session, id[%lu].", session_id);
            std::unique_lock<std::mutex> lock(session_mutex_);
            int32_t ret = GetOrCreateKernelCache<T>(session_kernel_cache_, session_id, sess_flag, kernel);
            if (ret != 0) {
                return ret;
            }
        } else {
            KERNEL_LOG_DEBUG("SessionCache KernelCache from stream, id[%lu].", stream_id);
            std::unique_lock<std::mutex> lock(stream_mutex_);
            int32_t ret = GetOrCreateKernelCache<T>(stream_kernel_cache_, stream_id, sess_flag, kernel);
            if (ret != 0) {
                return ret;
            }
        }
        return kernel->RunKernel(param);
    }

private:
    SessionCache() = default;
    ~SessionCache() = default;
    SessionCache(const SessionCache&) = delete;
    SessionCache(SessionCache&&) = delete;
    SessionCache& operator=(const SessionCache&) = delete;
    SessionCache& operator=(SessionCache&&) = delete;

    template <class T>
    int32_t GetOrCreateKernelCache(std::map<uint64_t, std::shared_ptr<KernelCache<C>>>& kernel_map, uint64_t id,
                                   bool sess_flag, std::shared_ptr<KernelCache<C>>& kernel)
    {
        auto iter = kernel_map.find(id);
        if (iter != kernel_map.end()) {
            KERNEL_LOG_DEBUG("Get kernel from cache success, id[%lu].", id);
            kernel = iter->second;
        } else {
            if (sess_flag) {
                if (session_id_cache_.size() > kSessionCacheNum) {
                    uint64_t del_key = session_id_cache_.back();
                    KERNEL_LOG_DEBUG("Session cache full, del id:%lu.", del_key);
                    session_id_cache_.pop_back();
                    auto del_iter = kernel_map.find(del_key); // just session_kernel_cache_
                    if (del_iter != kernel_map.end()) {
                        kernel_map.erase(del_iter);
                    }
                }
                session_id_cache_.push_front(id);
            }

            KernelCache<C>* cache = new (std::nothrow) T();
            if (cache == nullptr) {
                KERNEL_LOG_DEBUG("Create kernel cache failed, id[%lu].", id);
                return -1;
            }
            kernel = std::shared_ptr<KernelCache<C>>(cache);
            int32_t ret = kernel->Init(sess_flag);
            if (ret != 0) {
                return ret;
            }
            kernel_map.insert(std::make_pair(id, kernel));
            KERNEL_LOG_DEBUG("Create kernel cache, id[%lu].", id);
        }
        return 0;
    }

    std::mutex stream_mutex_;
    std::map<uint64_t, std::shared_ptr<KernelCache<C>>> stream_kernel_cache_; // key is stream id
    std::mutex session_mutex_;
    std::map<uint64_t, std::shared_ptr<KernelCache<C>>> session_kernel_cache_; // key is session id
    std::list<uint64_t> session_id_cache_;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_SESSION_CACHE_H
