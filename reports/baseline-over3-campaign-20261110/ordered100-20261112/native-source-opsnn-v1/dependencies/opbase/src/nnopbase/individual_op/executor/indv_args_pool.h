/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_ARGS_POOL_H_
#define INDV_ARGS_POOL_H_

#include <list>
#include "indv_executor.h"
#include "indv_tilingcontext_builder.h"

namespace nnopbase {
class ArgsPool {
public:
    static ArgsPool& GetInstance();
    ~ArgsPool();
    void Finalize();

    // 通过executor的ownArgs，在这两个接口间传递了inputKey, keyLen, enableCache, seed
    bool MatchArgs(NnopbaseExecutor* executor);
    aclnnStatus CreateArgs(NnopbaseExecutor* executor);
    inline void ReleaseArgs(NnopbaseExecutorArgs* const args)
    {
        if (args == nullptr) {
            return;
        }
        const std::lock_guard<std::mutex> lk(mutex);
        ReleaseFixedCache(args);
        args->isVisit = false;
    }

    void FixCache(NnopbaseExecutorArgs* const args);

    void Put(NnopbaseExecutorArgs* const args);

private:
    ArgsPool() = default;

    // 在调用端保证此处传入的args一定是缓存中存在的
    inline void Get(NnopbaseExecutorArgs* const args)
    {
        const auto& it = argsCache.find(args);
        if (it != argsCache.end()) {
            cacheList.splice(cacheList.begin(), cacheList, it->second);
        }
    }
    void ReleaseFixedCache(NnopbaseExecutorArgs* const args);
    bool IsArgsMatch(NnopbaseExecutorArgs* const args, NnopbaseExecutor* executor);
    void EraseArgs(NnopbaseExecutorArgs* const tmp);

    static size_t GetCacheSizeLimit();

    static size_t maxCacheNum;
    std::mutex mutex;
    std::list<NnopbaseExecutorArgs*> cacheList;
    std::unordered_map<NnopbaseExecutorArgs*, std::list<NnopbaseExecutorArgs*>::iterator> argsCache;
    std::unordered_map<size_t, std::list<NnopbaseExecutorArgs*>> argsMap;
    std::unordered_map<size_t, std::vector<NnopbaseExecutorArgs*>> fixedCacheMap;
};
} // namespace nnopbase
#endif