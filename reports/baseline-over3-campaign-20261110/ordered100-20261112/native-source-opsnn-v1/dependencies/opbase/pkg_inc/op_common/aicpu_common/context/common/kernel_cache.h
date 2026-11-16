/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_KERNEL_CACHE_H
#define AICPU_CONTEXT_COMMON_KERNEL_CACHE_H

#include <cstdint>

#include <list>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "log.h"
#include "device_cpu_kernel.h"

namespace aicpu {
template <class T>
class KernelCache {
public:
    KernelCache() : sess_flag_(false), capacity_(1) {}
    virtual ~KernelCache() = default;

    /*
     * Init kernel cache.
     * @param sess_flag: whether it's a session scene, false need to support LRU
     * algorithm
     * @return int32_t: 0 indicates success, while the others fail
     */
    int32_t Init(bool sess_flag)
    {
        sess_flag_ = sess_flag;
        return InitParameter();
    }

    /*
     * run kernel.
     * @param param: kernel context
     * @return int32_t: 0 indicates success, whilWe the others fail
     */
    virtual int32_t RunKernel(void* param) = 0;

    /*
     * run kernel with blockDimInfo.
     * @param param: kernel context and kernel context and blk_dim_info
     * @return int32_t: 0 indicates success, whilWe the others fail
     */
    virtual int32_t RunCpuKernelWithBlock(void* param, struct BlkDimInfo* blk_dim_info) = 0;
    /*
     * get kernel cache, the lru algorithm is supported in non-session scenarios
     * @param key: kernel id
     * @return std::shared_ptr<T>: cache content shared pointer
     */
    std::shared_ptr<T> GetCache(uint64_t key)
    {
        KERNEL_LOG_INFO("GetCache begin, key[%lu].", key);
        std::unique_lock<std::mutex> lock(kernel_mutex_);

        auto it = kernel_cache_iter_.find(key);
        if (it != kernel_cache_iter_.end()) {
            KERNEL_LOG_INFO("GetCache success, key[%lu].", key);
            auto shared = it->second->second;
            if (!sess_flag_) {
                auto pair_iter = it->second;
                std::pair<uint64_t, std::shared_ptr<T>> pair = *pair_iter;
                kernel_cache_.erase(pair_iter);
                kernel_cache_.push_front(pair);
                kernel_cache_iter_[key] = kernel_cache_.begin();
            }

            return shared;
        }

        return nullptr;
    }

    /*
     * set kernel cache, the lru algorithm is supported in non-session scenarios
     * @param key: kernel id
     * @param value: cache content
     */
    void SetCache(uint64_t key, std::shared_ptr<T> value)
    {
        KERNEL_LOG_INFO("SetCache begin, key[%lu].", key);
        std::unique_lock<std::mutex> lock(kernel_mutex_);
        auto iter = kernel_cache_iter_.find(key);
        if (iter != kernel_cache_iter_.end()) {
            KERNEL_LOG_INFO("SetCache update cache, key[%lu].", key);
            auto pair_iter = iter->second;
            pair_iter->second = value;
            if (!sess_flag_) {
                std::pair<uint64_t, std::shared_ptr<T>> pair = *pair_iter;
                kernel_cache_.erase(pair_iter);
                kernel_cache_.push_front(pair);
                kernel_cache_iter_[key] = kernel_cache_.begin();
            }
        } else {
            std::pair<uint64_t, std::shared_ptr<T>> pair = std::make_pair(key, value);
            if (capacity_ < kernel_cache_.size()) {
                uint64_t del_key = kernel_cache_.back().first;
                KERNEL_LOG_INFO("Cache is full, pop last element, capacity[%u], k[%lu], delete "
                                "k[%lu], kernel cache size[%zu], cache iter size[%zu].",
                                capacity_, key, del_key, kernel_cache_.size(), kernel_cache_iter_.size());
                kernel_cache_.pop_back();
                auto del_iter = kernel_cache_iter_.find(del_key);
                if (del_iter != kernel_cache_iter_.end()) {
                    kernel_cache_iter_.erase(del_iter);
                }
            }
            KERNEL_LOG_INFO("SetCache success, key[%lu].", key);
            kernel_cache_.push_front(pair);
            kernel_cache_iter_[key] = kernel_cache_.begin();
        }
    }

    /*
     * get session flag, true means session scene
     * @return bool: whether it's a session scene
     */
    bool GetSessionFlag() const { return sess_flag_; }

    /*
     * get kernel cache capacity
     * @return uint32_t: lru capacity
     */
    uint32_t GetCapacity() { return capacity_; }

    /*
     * set kernel cache capacity
     * @param capacity: lru capacity
     */
    void SetCapacity(uint32_t capacity) { capacity_ = capacity; }

    /*
     * get all kernel cache
     * @return std::list<std::pair<uint64_t, std::shared_ptr<T>>>: all cache,
     * pair<kernel id, cahce>
     */
    std::list<std::pair<uint64_t, std::shared_ptr<T>>> GetAllKernelCache() { return kernel_cache_; }

protected:
    virtual int32_t InitParameter() = 0;

private:
    KernelCache(const KernelCache&) = delete;
    KernelCache(KernelCache&&) = delete;
    KernelCache& operator=(const KernelCache&) = delete;
    KernelCache& operator=(KernelCache&&) = delete;

    bool sess_flag_;    // whether it's a session scene, false need to support LRU
    uint32_t capacity_; // lru capacity
    std::mutex kernel_mutex_;
    std::list<std::pair<uint64_t, std::shared_ptr<T>>> kernel_cache_; // all kernel cache, key is kernel id
    std::unordered_map<uint64_t,
                       typename std::list<std::pair<uint64_t, std::shared_ptr<T>>>::iterator> // iterator of kernel
                                                                                              // cahce, key is kernel id
                                                                                                  kernel_cache_iter_;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_KERNEL_CACHE_H
