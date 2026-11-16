/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_args_pool.h"
#include "indv_cache_key_builder.h"
#include "mmpa/mmpa_api.h"
#include "utils/thread_var_container.h"

namespace nnopbase {
namespace {
// cache中的总数，默认1万，用户可控，上限1000万
constexpr size_t NNOPBASE_CACHE_ARGS_NUM = 10000U;
constexpr size_t NNOPBASE_MAX_CACHE_NUM = 10000000U;
constexpr size_t HASH_FACTOR = 2U;
} // namespace

size_t ArgsPool::maxCacheNum = ArgsPool::GetCacheSizeLimit();

size_t ArgsPool::GetCacheSizeLimit()
{
    const char_t* cacheLimit = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ACLNN_CACHE_LIMIT, cacheLimit);
    size_t c = NNOPBASE_CACHE_ARGS_NUM;
    if (cacheLimit != nullptr) {
        try {
            c = std::stoull(cacheLimit);
        } catch (const std::exception& e) {
            OP_LOGW("Env variable ACLNN_CACHE_LIMIT[%s] is invalid! must be a number!", cacheLimit);
        }
    }
    c = std::min(c, NNOPBASE_MAX_CACHE_NUM);
    OP_LOGI("Cachelimit is %zu", c);
    GetInstance().argsCache.reserve(c * HASH_FACTOR);
    GetInstance().argsMap.reserve(c * HASH_FACTOR);
    return c;
}

ArgsPool& ArgsPool::GetInstance()
{
    static ArgsPool args;
    return args;
}

ArgsPool::~ArgsPool() { Finalize(); }

void ArgsPool::Finalize()
{
    for (auto& iter : argsMap) {
        for (auto& args : iter.second) {
            if (args != nullptr) {
                delete args;
                args = nullptr;
            }
        }
    }
    fixedCacheMap.clear();
    argsMap.clear();
    cacheList.clear();
    argsCache.clear();
}

bool ArgsPool::IsArgsMatch(NnopbaseExecutorArgs* const args, NnopbaseExecutor* executor)
{
    if (args->isVisit) {
        OP_LOGI("Op %s seed %zu args %p is visited.", executor->opType, args->seed, args);
        return false;
    }
    if (args->keyLen != executor->ownArgs.keyLen) {
        OP_LOGI("Op %s seed %zu key len is not equal, cache len %zu, input len %zu.", executor->opType, args->seed,
                args->keyLen, executor->ownArgs.keyLen);
        return false;
    }
    if (memcmp(args->inputKey.data(), executor->ownArgs.inputKey.data(), args->keyLen) == 0) {
        executor->args = args;
        executor->hasTiling = (!args->binInfo->isStaticShape);
        executor->isCachedArgs = true;
        args->isVisit = true;
        Get(executor->args);
        OP_LOGI("Op %s match args cache successfully, seed is %zu, key len is %zu.", executor->opType, args->seed,
                args->keyLen);
        return true;
    }
    return false;
}

bool ArgsPool::MatchArgs(NnopbaseExecutor* executor)
{
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheStart);
    if (maxCacheNum == 0U) {
        executor->ownArgs.enableCache = false;
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
        return false;
    }
    if (!executor->matchArgsV2) {
        Indv::CacheKeyBuilder::GenerateCacheArgsKeyV1(executor);
    }
    {
        const std::lock_guard<std::mutex> lk(mutex);
        const auto& iter = argsMap.find(executor->ownArgs.seed);
        if (iter != argsMap.end()) {
            OP_LOGI("Op %s seed %zu args num is %zu.", executor->opType, executor->ownArgs.seed, iter->second.size());
            for (auto& args : iter->second) {
                if (IsArgsMatch(args, executor)) {
                    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
                    return true;
                }
            }
        }
    }
    OP_LOGI("Op %s cache miss, seed is %zu, key len is %zu.", executor->opType, executor->ownArgs.seed,
            executor->ownArgs.keyLen);
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
    return false;
}

aclnnStatus ArgsPool::CreateArgs(NnopbaseExecutor* executor)
{
    if (executor->ownArgs.enableCache) {
        auto args = std::make_unique<NnopbaseExecutorArgs>();
        NNOPBASE_ASSERT_NOTNULL_RETVAL(args);
        {
            const std::lock_guard<std::mutex> lk(mutex);
            (void)argsMap[executor->ownArgs.seed].emplace_back(args.release());
            executor->args = argsMap[executor->ownArgs.seed].back();
            executor->args->isVisit = true;
            RecordNnopbaseTime(executor, NnopbaseTimeIdx::kCreateCacheEnd);
            Put(executor->args);
        }
        executor->args->keyLen = executor->ownArgs.keyLen;
        executor->args->seed = executor->ownArgs.seed;
        if (executor->args->keyLen > executor->args->inputKey.size()) {
            executor->args->inputKey.resize(executor->args->keyLen);
        }
        NNOPBASE_ASSERT_TRUE_RETVAL(memcpy_s(executor->args->inputKey.data(), executor->ownArgs.keyLen,
                                             executor->ownArgs.inputKey.data(), executor->ownArgs.keyLen) == EOK);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSaveCachedTensor(&executor->args->inputs, &executor->ownArgs.inputs, true));
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseSaveCachedTensor(&executor->args->outputs, &executor->ownArgs.outputs, false));
        NnopbaseSaveUnContiguousTensors(&executor->args->inputs, &executor->ownArgs.inputs);
    } else {
        executor->args = &executor->ownArgs;
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kCreateCacheEnd);
    }
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kAgingCacheEnd);
    return OK;
}

void ArgsPool::EraseArgs(NnopbaseExecutorArgs* const tmp)
{
    auto& argsList = argsMap[tmp->seed];
    for (auto it = argsList.begin(); it != argsList.end(); it++) {
        if (*it == tmp) {
            OP_LOGI("The number of args %zu cached has reached %zu, delete the oldest args %p seed %zu.",
                    argsCache.size(), maxCacheNum, tmp, tmp->seed);
            (void)argsList.erase(it);
            if (argsList.empty()) {
                (void)argsMap.erase(tmp->seed);
            }
            break;
        }
    }
    (void)argsCache.erase(tmp);
    delete tmp;
    return;
}

// 在调用端保证一定传入新的args进来，make出来的args才会调用Put
void ArgsPool::Put(NnopbaseExecutorArgs* const args)
{
    while ((argsCache.size() >= maxCacheNum) && (!cacheList.empty())) {
        const auto& tmp = cacheList.back();
        if (tmp->isVisit) {
            OP_LOGW("The number of args cached has reached %zu, but the oldest args %p seed %zu is in use! "
                    "Can't delete args!",
                    maxCacheNum, tmp, tmp->seed);
            break;
        }
        EraseArgs(tmp);
        cacheList.pop_back();
    }
    cacheList.emplace_front(args);
    argsCache[args] = cacheList.begin();
}

void ArgsPool::FixCache(NnopbaseExecutorArgs* const args)
{
    if (args == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> lk(mutex);
    auto& argsList = argsMap[args->seed];
    for (auto it = argsList.begin(); it != argsList.end(); it++) {
        if (*it == args) {
            (void)argsList.erase(it);
            const auto& argsPos = argsCache.find(args);
            if (argsPos != argsCache.end()) {
                cacheList.erase(argsPos->second);
            }
            if (argsList.empty()) {
                (void)argsMap.erase(args->seed);
            }
            break;
        }
    }
    (void)argsCache.erase(args);
    fixedCacheMap[args->seed].push_back(args);
    OP_LOGI("Fix args cache successfully, current fixed cache pool size for seed %zu is %zu", args->seed,
            fixedCacheMap[args->seed].size());
    return;
}

void ArgsPool::ReleaseFixedCache(NnopbaseExecutorArgs* const args)
{
    auto cacheListIter = fixedCacheMap.find(args->seed);
    if (cacheListIter == fixedCacheMap.cend()) {
        return;
    }
    auto& fixedCacheList = cacheListIter->second;
    for (auto it = fixedCacheList.begin(); it != fixedCacheList.end(); it++) {
        if (*it == args) {
            Put(args);
            fixedCacheList.erase(it);
            break;
        }
    }
    if (fixedCacheList.empty()) {
        (void)fixedCacheMap.erase(args->seed);
    }
    OP_LOGI("Release fixed args cache successfully, current fixed cache pool size is %zu", fixedCacheMap.size());
    return;
}
} // namespace nnopbase