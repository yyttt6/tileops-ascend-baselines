/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H
#define AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H

#include <functional>
#include <memory>
#include <mutex>
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

namespace aicpu {
using SharderWork = std::function<void(int64_t, int64_t)>;

class EigenThreadPoolEmbedding {
public:
    static EigenThreadPoolEmbedding* GetInstance();

    /*
     * ParallelFor shards the "total" units of work.
     */
    void ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork& work) const;

private:
    EigenThreadPoolEmbedding() = default;
    ~EigenThreadPoolEmbedding() = default;

    EigenThreadPoolEmbedding(const EigenThreadPoolEmbedding&) = delete;
    EigenThreadPoolEmbedding(EigenThreadPoolEmbedding&&) = delete;
    EigenThreadPoolEmbedding& operator=(const EigenThreadPoolEmbedding&) = delete;
    EigenThreadPoolEmbedding& operator=(EigenThreadPoolEmbedding&&) = delete;

    static std::mutex mutex_; // protect init_flag_
    static bool init_flag_;   // true means initialized
    static int32_t core_num_; // the number of CPU cores that can be used by users
    static std::unique_ptr<Eigen::ThreadPool> eigen_threadpool_;
    static std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
};
}; // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H
