/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * MurmurHash3 was written by Austin Appleby, and is placed in the public
 * domain. The author hereby disclaims copyright to this source code.
 */

#include "opdev/op_cache.h"
#include <sstream>
#include <atomic>
#include <unistd.h>
#include "opdev/op_dfx.h"
#include "opdev/fast_vector.h"
#include "kernel_utils.h"
#include "op_dfx_internal.h"
#include "op_cache_internal.h"
#include "op_feature_internal.h"
#include "thread_local_context.h"
#include "opdev/op_cache_container.h"
#include "lock_free_queue.h"
#include "bridge_dfx.h"

using namespace std;
namespace op {
namespace internal {
constexpr int64_t SLEEP_ONE_SECOND = 1000000;
constexpr int64_t CACHE_DESTRUCT_MAX_WAIT_TIME = 60; // 1 min
using char_t = char;

class OpExecCacheManager {
public:
    OpExecCacheManager();

    ~OpExecCacheManager();

    void InitCache();
    void ClearCacheManually();

    OpExecCache* GetOpExecCache(uint64_t hash);
    OpExecCache* GetOpExecCache(OpCacheKey& key);

    size_t GetCacheSizeLimit();

    bool AddOpExecCache(OpExecCache* exec);
    void RemoveOpExecCache(OpExecCache* exec);

    void ShrinkCache(const size_t num, ListHead* shrinkList);

    void Start();

    void SubmitGcTask(ListHead* c);

    void IncreaseUseCount() { useCount_++; }

    void DecreaseUseCount() { useCount_--; }

private:
    void WaitCacheCompleteUse();
    void DeleteCache1();

    size_t cacheLimit_;
    std::mutex lock_;
    std::unordered_map<uint64_t, OpExecCache*> cache_;
    OpCacheContainer<OpCacheKey, OpCacheValue, OpCacheKeyHash, OpCacheKeyEqual> cache2_;
    // for gc
    bool gcInitialize_{false};
    std::atomic<bool> threadStop_{false};
    std::thread consumer_;
    LockFreeQueue<ListHead*> gcQueue_;
    std::atomic<int64_t> useCount_{0};
};

thread_local char g_hashBuf[K_HASH_BUF_SIZE];
thread_local uint64_t g_hashOffset = 0;

thread_local char g_cacheBuf[K_CACHE_BUF_SIZE];
thread_local OpExecCacheManager g_opExecCacheManager;

std::atomic<bool> g_enableOpCacheCount{true};

class OpCacheThreadLocalData {
public:
    char* hashBuf;
    uint64_t& hashOffset;
    unordered_map<const aclTensor*, uint64_t> tensorLabelMap;
    uint64_t tensorLabelNum;
    std::vector<int64_t> tensorLabelList;

    OpThreadLocalContext& threadLocalContext;

    OpCacheThreadLocalData()
        : hashBuf(g_hashBuf),
          hashOffset(g_hashOffset),
          tensorLabelNum(0),
          threadLocalContext(op::internal::GetThreadLocalContext())
    {}
};

thread_local OpCacheThreadLocalData g_opCacheTlsData;

constexpr int REMAIND_ONE = 1;
constexpr int REMAIND_TWO = 2;
constexpr int REMAIND_THREE = 3;
constexpr int REMAIND_FOUR = 4;
constexpr int REMAIND_FIVE = 5;
constexpr int REMAIND_SIX = 6;
constexpr int REMAIND_SEVEN = 7;
constexpr int REMAIND_EIGHT = 8;
constexpr int REMAIND_NINE = 9;
constexpr int REMAIND_TEN = 10;
constexpr int REMAIND_ELEVEN = 11;
constexpr int REMAIND_TWELVE = 12;
constexpr int REMAIND_THIRTEEN = 13;
constexpr int REMAIND_FORTEEN = 14;
constexpr int REMAIND_FIFTEEN = 15;

constexpr int FIRST_BTYE = 0;
constexpr int SECOND_BTYE = 1;
constexpr int THIRD_BTYE = 2;
constexpr int FORTH_BTYE = 3;
constexpr int FIFTH_BTYE = 4;
constexpr int SIXTH_BTYE = 5;
constexpr int SEVENTH_BTYE = 6;
constexpr int EIGHTH_BTYE = 7;
constexpr int NINTH_BTYE = 8;
constexpr int TENTH_BTYE = 9;
constexpr int ELEVENTH_BTYE = 10;
constexpr int TWELVETH_BTYE = 11;
constexpr int THIRTEENTH_BTYE = 12;
constexpr int FORTEENTH_BTYE = 13;
constexpr int FIFTEENTH_BTYE = 14;

constexpr int SECOND_BTYPE_SHIFT = 8;
constexpr int THIRD_BTYPE_SHIFT = 16;
constexpr int FORTH_BTYPE_SHIFT = 24;
constexpr int FIFTH_BTYPE_SHIFT = 32;
constexpr int SIXTH_BTYPE_SHIFT = 40;
constexpr int SEVENTH_BTYPE_SHIFT = 48;
constexpr int EIGHTH_BTYPE_SHIFT = 56;
constexpr int FMIX_SHIFT = 33;
constexpr int8_t ROTL_CONSTANT = 64;
constexpr int8_t K1_SHIFT = 31;
constexpr int8_t K2_SHIFT = 33;
constexpr int8_t H1_SHIFT = 27;
constexpr int8_t H2_SHIFT = 31;
constexpr uint64_t MUL_NUM = 5;

#ifdef __cplusplus
extern "C" {
#endif
void ResetCacheThreadLocal()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.cacheHashKey_ = nullptr;
    tlsData->threadLocalContext.cacheHashKeyLen_ = 0;
    tlsData->threadLocalContext.hashKey_ = 0;
    tlsData->threadLocalContext.usePTAHash_ = false;
    tlsData->hashOffset = 0;
    tlsData->threadLocalContext.cachedTensorListSize_ = 0;
    tlsData->threadLocalContext.cachedStorageListSize_ = 0;
    tlsData->tensorLabelMap.clear();
    tlsData->tensorLabelNum = 0;
    tlsData->tensorLabelList.clear();
    tlsData->threadLocalContext.opConfigInfo_.usePcieAddr = false;
}

void UnInitPTACacheThreadLocal()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.usePTAHash_ = false;
}

void InitPTACacheThreadLocal()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.cacheHashKey_ = nullptr;
    tlsData->threadLocalContext.cacheHashKeyLen_ = 0;
    tlsData->threadLocalContext.hashKey_ = 0;
    tlsData->threadLocalContext.usePTAHash_ = true;
    tlsData->hashOffset = 0;
    tlsData->threadLocalContext.cachedTensorListSize_ = 0;
    tlsData->threadLocalContext.cachedStorageListSize_ = 0;
    tlsData->tensorLabelMap.clear();
    tlsData->tensorLabelNum = 0;
    tlsData->tensorLabelList.clear();
    tlsData->threadLocalContext.opConfigInfo_.usePcieAddr = false;
}

void AddTensorAddrToCachedList(void* addr)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    auto& tlsCachedTensorList = tlsData->threadLocalContext.cachedTensorList_;
    auto& tlsCachedTensorListSize = tlsData->threadLocalContext.cachedTensorListSize_;
    if (tlsCachedTensorListSize >= tlsCachedTensorList.size()) {
        tlsCachedTensorList.push_back(addr);
    } else {
        tlsCachedTensorList.at(tlsCachedTensorListSize) = addr;
    }
    tlsCachedTensorListSize++;
    if (op::internal::IsPcieThroughEnabled() && op::internal::IsTensorAddrInPcieRange(addr)) {
        tlsData->threadLocalContext.opConfigInfo_.usePcieAddr = true;
    }
}

void SetPTAHashKey(uint64_t hash)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.hashKey_ = hash;
};

void SetPTACacheHashKey(uint8_t* key, size_t len)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.cacheHashKey_ = key;
    tlsData->threadLocalContext.cacheHashKeyLen_ = len;
}

#ifdef __cplusplus
}
#endif

void* GetCacheBuf() { return static_cast<void*>(g_cacheBuf); }

void InitExecutorCacheThreadLocal()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (!tlsData->threadLocalContext.usePTAHash_) {
        tlsData->threadLocalContext.cacheHashKey_ = nullptr;
        tlsData->threadLocalContext.cacheHashKeyLen_ = 0;
        tlsData->threadLocalContext.hashKey_ = 0;
    }
    tlsData->hashOffset = 0;
    tlsData->threadLocalContext.cachedTensorListSize_ = 0;
    tlsData->threadLocalContext.cachedStorageListSize_ = 0;
    tlsData->tensorLabelMap.clear();
    tlsData->tensorLabelNum = 0;
    tlsData->tensorLabelList.clear();
    tlsData->threadLocalContext.opConfigInfo_.usePcieAddr = false;
}

static void AddAclTensorToCachedList(const aclTensor* tensor, OpCacheThreadLocalData* tlsData)
{
    auto& tlsCachedStorageList = tlsData->threadLocalContext.cachedStorageList_;
    auto& tlsCachedStorageListSize = tlsData->threadLocalContext.cachedStorageListSize_;
    if (tlsCachedStorageListSize >= tlsCachedStorageList.size()) {
        tlsCachedStorageList.push_back(tensor->GetStorage());
    } else {
        tlsCachedStorageList.at(tlsCachedStorageListSize) = tensor->GetStorage();
    }
    tlsCachedStorageListSize++;

    auto& tlsCachedTensorList = tlsData->threadLocalContext.cachedTensorList_;
    auto& tlsCachedTensorListSize = tlsData->threadLocalContext.cachedTensorListSize_;
    if (tlsCachedTensorListSize >= tlsCachedTensorList.size()) {
        tlsCachedTensorList.push_back(tensor->GetStorageAddr());
    } else {
        tlsCachedTensorList.at(tlsCachedTensorListSize) = tensor->GetStorageAddr();
    }
    tlsCachedTensorListSize++;

    auto it = tlsData->tensorLabelMap.find(tensor);
    if (it == tlsData->tensorLabelMap.end()) {
        uint64_t index = tlsData->tensorLabelNum++;
        tlsData->tensorLabelMap[tensor] = index;
        tlsData->tensorLabelList.push_back(index);
    } else {
        tlsData->tensorLabelList.push_back(it->second);
    }
    if (op::internal::IsPcieThroughEnabled() && op::internal::IsTensorAddrInPcieRange(tensor->GetStorageAddr())) {
        tlsData->threadLocalContext.opConfigInfo_.usePcieAddr = true;
    }
}

inline uint64_t Rotl64(uint64_t x, int8_t r) { return (x << r) | (x >> (ROTL_CONSTANT - r)); }

inline uint64_t GetBlock64(const uint64_t* p, int i) { return p[i]; }

inline uint64_t Fmix64(uint64_t k)
{
    // 0xff51afd7ed558ccd and 0xc4ceb9fe1a85ec53 are carefully selected constants to allow
    // hash values to be more evenly distributed in 64-bit space after multiplication.
    k ^= k >> FMIX_SHIFT;
    k *= 0xff51afd7ed558ccdLLU;
    k ^= k >> FMIX_SHIFT;
    k *= 0xc4ceb9fe1a85ec53LLU;
    k ^= k >> FMIX_SHIFT;

    return k;
}

inline uint64_t MurmurHash(const void* key, const int len, const uint32_t seed = 0xdeadb0d7)
{
    const uint8_t* data = PtrCastTo<const uint8_t>(key);
    // the length of each block is 16 bytes
    const int nblocks = len / 16;
    uint64_t h1 = seed;
    uint64_t h2 = seed;

    // 0x87c37b91114253d5 and 0x4cf5ad432745937f are carefully selected constants to
    // blocking and obfuscation of input data
    const uint64_t c1 = 0x87c37b91114253d5LLU;
    const uint64_t c2 = 0x4cf5ad432745937fLLU;

    const uint64_t* blocks = PtrCastTo<const uint64_t>(data);

    for (int i = 0; i < nblocks; i++) {
        int evenNum = 2;
        int oddNum = 1;
        uint64_t k1 = GetBlock64(blocks, i * evenNum);
        uint64_t k2 = GetBlock64(blocks, i * evenNum + oddNum);

        k1 *= c1;
        k1 = Rotl64(k1, K1_SHIFT);
        k1 *= c2;
        h1 ^= k1;

        h1 = Rotl64(h1, H1_SHIFT);
        h1 += h2;
        // increase randomness by mul by 5 and adding a constant
        h1 = h1 * MUL_NUM + 0x52dce729;

        k2 *= c2;
        k2 = Rotl64(k2, K2_SHIFT);
        k2 *= c1;
        h2 ^= k2;

        h2 = Rotl64(h2, H2_SHIFT);
        h2 += h1;
        // increase randomness by mul by 5 and adding a constant
        h2 = h2 * MUL_NUM + 0x38495ab5;
    }

    // the length of each block is 16 bytes
    const uint8_t* tail = PtrCastTo<const uint8_t>(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    // because the size of a block is 16, different offsets are calculated for tail blocks
    // for different sizes
    switch (len & REMAIND_FIFTEEN) {
        case REMAIND_FIFTEEN:
            k2 ^= (static_cast<uint64_t>(tail[FIFTEENTH_BTYE])) << SEVENTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_FORTEEN:
            k2 ^= (static_cast<uint64_t>(tail[FORTEENTH_BTYE])) << SIXTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_THIRTEEN:
            k2 ^= (static_cast<uint64_t>(tail[THIRTEENTH_BTYE])) << FIFTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_TWELVE:
            k2 ^= (static_cast<uint64_t>(tail[TWELVETH_BTYE])) << FORTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_ELEVEN:
            k2 ^= (static_cast<uint64_t>(tail[ELEVENTH_BTYE])) << THIRD_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_TEN:
            k2 ^= (static_cast<uint64_t>(tail[TENTH_BTYE])) << SECOND_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_NINE:
            k2 ^= (static_cast<uint64_t>(tail[NINTH_BTYE]));
            k2 *= c2;
            k2 = Rotl64(k2, K2_SHIFT);
            k2 *= c1;
            h2 ^= k2;
            [[fallthrough]];
        case REMAIND_EIGHT:
            k1 ^= (static_cast<uint64_t>(tail[EIGHTH_BTYE])) << EIGHTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_SEVEN:
            k1 ^= (static_cast<uint64_t>(tail[SEVENTH_BTYE])) << SEVENTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_SIX:
            k1 ^= (static_cast<uint64_t>(tail[SIXTH_BTYE])) << SIXTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_FIVE:
            k1 ^= (static_cast<uint64_t>(tail[FIFTH_BTYE])) << FIFTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_FOUR:
            k1 ^= (static_cast<uint64_t>(tail[FORTH_BTYE])) << FORTH_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_THREE:
            k1 ^= (static_cast<uint64_t>(tail[THIRD_BTYE])) << THIRD_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_TWO:
            k1 ^= (static_cast<uint64_t>(tail[SECOND_BTYE])) << SECOND_BTYPE_SHIFT;
            [[fallthrough]];
        case REMAIND_ONE:
            k1 ^= (static_cast<uint64_t>(tail[FIRST_BTYE]));
            k1 *= c1;
            k1 = Rotl64(k1, K1_SHIFT);
            k1 *= c2;
            h1 ^= k1;
            [[fallthrough]];
        default:
            break;
    };

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = Fmix64(h1);
    h2 = Fmix64(h2);

    h1 += h2;
    h2 += h1;
    return h2;
}

static inline bool CheckHashBufCapacity(uint64_t& hashOffset, size_t addSize)
{
    if (hashOffset + addSize > K_HASH_BUF_SIZE) {
        hashOffset = K_INVALID_HASH_OFFSET;
        OP_LOGW("Hash buffer size is overflow, cur size: %lu, add size: %zu, capacity: %lu", hashOffset, addSize,
                K_HASH_BUF_SIZE);
        return false;
    }
    return true;
}

static inline void OpCacheAddSeparator(char* hashBuf, uint64_t& hashOffset)
{
    hashBuf[hashOffset] = ',';
    hashOffset += 1;
}

static inline void OpCacheAdd1Byte(const void* buf, char* hashBuf, uint64_t& hashOffset)
{
    *PtrCastTo<uint8_t>(hashBuf + hashOffset) = *PtrCastTo<uint8_t>(buf);
    hashOffset += sizeof(uint8_t);
}

static inline void OpCacheAdd4Byte(const void* buf, char* hashBuf, uint64_t& hashOffset)
{
    *PtrCastTo<uint32_t>(hashBuf + hashOffset) = *PtrCastTo<uint32_t>(buf);
    hashOffset += sizeof(uint32_t);
}

static inline void OpCacheAdd8Byte(const void* buf, char* hashBuf, uint64_t& hashOffset)
{
    *PtrCastTo<uint64_t>(hashBuf + hashOffset) = *PtrCastTo<uint64_t>(buf);
    hashOffset += sizeof(uint64_t);
}

template <typename T>
void AddParamToBufV2(const T* addr, uint64_t size, char* hashBuf, uint64_t& hashOffset)
{
    if (hashOffset + size * sizeof(T) > K_HASH_BUF_SIZE) {
        hashOffset = K_INVALID_HASH_OFFSET;
        return;
    }
    OP_CHECK(memcpy_s(hashBuf + hashOffset, K_HASH_BUF_SIZE - hashOffset, addr, size * sizeof(T)) == EOK,
             OP_LOGW("Failed to memcpy in op cache."),
             ;);
    hashOffset += size * sizeof(T);
};

void SetOpCacheKey(OpCacheKey& key)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (tlsData->threadLocalContext.usePTAHash_) {
        key.buf = tlsData->threadLocalContext.cacheHashKey_;
        key.len = tlsData->threadLocalContext.cacheHashKeyLen_;
        return;
    }
    char* hashBuf = tlsData->hashBuf;
    uint64_t& hashOffset = tlsData->hashOffset;
    auto& tensorLabelList = tlsData->tensorLabelList;
    if (CheckHashBufCapacity(hashOffset, sizeof(int64_t) * tensorLabelList.size()) == true) {
        for (size_t i = 0; i < tensorLabelList.size(); i++) {
            OpCacheAdd8Byte(&tensorLabelList[i], hashBuf, hashOffset);
        }
    }
    if (hashOffset == K_INVALID_HASH_OFFSET) {
        tlsData->threadLocalContext.cacheHashKey_ = nullptr;
        tlsData->threadLocalContext.cacheHashKeyLen_ = 0;
        return;
    }
    tlsData->threadLocalContext.cacheHashKey_ = PtrCastTo<uint8_t>(hashBuf);
    tlsData->threadLocalContext.cacheHashKeyLen_ = static_cast<size_t>(hashOffset);
    key.buf = PtrCastTo<uint8_t>(hashBuf);
    key.len = static_cast<size_t>(hashOffset);
}

void AddOpConfigInfoToBuf()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (CheckHashBufCapacity(tlsData->hashOffset, sizeof(ThreadCoreNum) + sizeof(uint8_t))) {
        ThreadCoreNum coreNum(tlsData->threadLocalContext.opConfigInfo_.aicNum_,
                              tlsData->threadLocalContext.opConfigInfo_.aivNum_);
        *PtrCastTo<ThreadCoreNum>(tlsData->hashBuf + tlsData->hashOffset) = coreNum;
        tlsData->hashOffset += sizeof(ThreadCoreNum);
        uint8_t deterministicLevel = tlsData->threadLocalContext.opConfigInfo_.deterministicLevel_;
        OpCacheAdd1Byte(&deterministicLevel, tlsData->hashBuf, tlsData->hashOffset);
    }
}

static void AddSeperator(OpCacheThreadLocalData* tlsData)
{
    if (tlsData->hashOffset + 1 > K_HASH_BUF_SIZE) {
        tlsData->hashOffset = K_INVALID_HASH_OFFSET;
        return;
    }
    tlsData->hashBuf[tlsData->hashOffset] = ',';
    tlsData->hashOffset += 1;
}

void AddSeperator()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    AddSeperator(tlsData);
}

static void AddParamToBuf(const aclTensor* tensor, OpCacheThreadLocalData* tlsData)
{
    if (tensor == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    char* hashBuf = tlsData->hashBuf;
    uint64_t& hashOffset = tlsData->hashOffset;
    AddAclTensorToCachedList(tensor, tlsData);

    const op::Shape& viewShape = tensor->GetViewShape();
    const op::Strides& strides = tensor->GetViewStrides();
    const op::Shape& storageShape = tensor->GetStorageShape();
    // view shape + separator + strides + separator + storage shape + separator
    size_t shapeStridesInfoSize = viewShape.GetDimNum() * sizeof(int64_t) + sizeof(uint8_t) +
                                  strides.size() * sizeof(int64_t) + sizeof(uint8_t) +
                                  storageShape.GetDimNum() * sizeof(int64_t) + sizeof(uint8_t);
    // data type + separator + offset + separator + format
    size_t otherInfoSize = sizeof(uint32_t) + sizeof(uint8_t) + sizeof(int64_t) + sizeof(uint8_t) + sizeof(uint32_t);
    if (CheckHashBufCapacity(hashOffset, shapeStridesInfoSize + otherInfoSize) == false) {
        return;
    }

    // view shape
    for (size_t i = 0; i < viewShape.GetDimNum(); i++) {
        OpCacheAdd8Byte(&viewShape[i], hashBuf, hashOffset);
    }
    OpCacheAddSeparator(hashBuf, hashOffset);
    // view stride
    for (size_t i = 0; i < strides.size(); i++) {
        OpCacheAdd8Byte(&strides[i], hashBuf, hashOffset);
    }
    OpCacheAddSeparator(hashBuf, hashOffset);
    // storage shape
    for (size_t i = 0; i < storageShape.GetDimNum(); i++) {
        OpCacheAdd8Byte(&storageShape[i], hashBuf, hashOffset);
    }
    OpCacheAddSeparator(hashBuf, hashOffset);
    // datatype
    op::DataType dataType = tensor->GetDataType();
    OpCacheAdd4Byte(&dataType, hashBuf, hashOffset);
    OpCacheAddSeparator(hashBuf, hashOffset);
    // offset
    int64_t offset = tensor->GetViewOffset();
    OpCacheAdd8Byte(&offset, hashBuf, hashOffset);
    OpCacheAddSeparator(hashBuf, hashOffset);
    // format
    op::Format viewFormat = tensor->GetViewFormat();
    OpCacheAdd4Byte(&viewFormat, hashBuf, hashOffset);
}

void AddParamToBuf(aclTensor* tensor)
{
    const aclTensor* constTensor = const_cast<const aclTensor*>(tensor);
    AddParamToBuf(constTensor);
}

void AddParamToBuf(const aclTensor* tensor)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    AddParamToBuf(tensor, tlsData);
};

void AddParamToBuf(const aclScalar* scalar)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (scalar == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    char* hashBuf = tlsData->hashBuf;
    uint64_t& hashOffset = tlsData->hashOffset;

    const size_t addSize = scalar->Size() + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t);
    if (CheckHashBufCapacity(hashOffset, addSize) == false) {
        return;
    }
    // scalar data
    OP_CHECK(memcpy_s(hashBuf + hashOffset, K_HASH_BUF_SIZE - hashOffset, scalar->GetData(), scalar->Size()) == EOK,
             OP_LOGW("Failed to memcpy in op cache."),
             ;);
    hashOffset += scalar->Size();
    OpCacheAddSeparator(hashBuf, hashOffset);
    // datatype
    op::DataType dataType = scalar->GetDataType();
    OpCacheAdd4Byte(&dataType, hashBuf, hashOffset);
    OpCacheAddSeparator(hashBuf, hashOffset);
};

void AddParamToBuf(aclScalar* scalar)
{
    const aclScalar* constScalar = scalar;
    AddParamToBuf(constScalar);
}

template <typename T>
static void OpCacheAddArray(const T* addr, size_t counter, char* hashBuf, uint64_t& hashOffset)
{
    if (CheckHashBufCapacity(hashOffset, counter * sizeof(T) + sizeof(uint64_t)) == false) {
        return;
    }
    OP_CHECK(memcpy_s(hashBuf + hashOffset, K_HASH_BUF_SIZE - hashOffset, addr, counter * sizeof(T)) == EOK,
             OP_LOGW("Failed to memcpy array data to cache."),
             ;);
    hashOffset += counter * sizeof(T);
    OpCacheAdd8Byte(&counter, hashBuf, hashOffset);
}

static inline void OpCacheAddCounter(const size_t counter, char* hashBuf, uint64_t& hashOffset)
{
    if (CheckHashBufCapacity(hashOffset, sizeof(uint64_t)) == false) {
        return;
    }
    OpCacheAdd8Byte(&counter, hashBuf, hashOffset);
}

void AddParamToBuf(const aclIntArray* value)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (value == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    OpCacheAddArray(value->GetData(), value->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const aclBoolArray* value)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (value == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    OpCacheAddArray(value->GetData(), value->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const aclFloatArray* value)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (value == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    OpCacheAddArray(value->GetData(), value->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const aclFp16Array* value)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (value == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    OpCacheAddArray(value->GetData(), value->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(aclTensorList* tensors)
{
    const aclTensorList* constTensors = const_cast<const aclTensorList*>(tensors);
    AddParamToBuf(constTensors);
}

void AddParamToBuf(const aclTensorList* tensors)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (tensors == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        const aclTensor* t = (*tensors)[i];
        AddParamToBuf(t, tlsData);
    }
    OpCacheAddCounter(tensors->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(aclScalarList* scalars)
{
    const aclScalarList* constScalars = const_cast<const aclScalarList*>(scalars);
    AddParamToBuf(constScalars);
}

void AddParamToBuf(const aclScalarList* scalars)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (scalars == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    for (uint64_t i = 0; i < scalars->Size(); i++) {
        const aclScalar* t = (*scalars)[i];
        AddParamToBuf(t);
    }
    OpCacheAddCounter(scalars->Size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const std::string& s)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    AddParamToBufV2(s.c_str(), s.size(), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const aclDataType dtype)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    AddParamToBufV2(&dtype, 1, tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(const char* c)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (c == nullptr) {
        AddSeperator(tlsData);
        return;
    }
    AddParamToBufV2(c, strlen(c), tlsData->hashBuf, tlsData->hashOffset);
    OpCacheAddCounter(strlen(c), tlsData->hashBuf, tlsData->hashOffset);
};

void AddParamToBuf(char* c)
{
    const char* constPtr = const_cast<const char*>(c);
    AddParamToBuf(constPtr);
};

void AddParamToBuf(){};

std::size_t OpCacheKeyHash::operator()(const OpCacheKey& key) const { return MurmurHash(key.buf, key.len); }

bool OpCacheKeyEqual::operator()(const OpCacheKey& lhs, const OpCacheKey& rhs) const { return lhs == rhs; }

OpExecCache* GetOpExecCache(uint64_t hash) { return g_opExecCacheManager.GetOpExecCache(hash); }

OpExecCache* GetOpExecCache(OpCacheKey& key) { return g_opExecCacheManager.GetOpExecCache(key); }

void RemoveExecCache(OpExecCache* exec) { g_opExecCacheManager.RemoveOpExecCache(exec); }

bool AddOpExecCache(OpExecCache* exec) { return g_opExecCacheManager.AddOpExecCache(exec); }

void* GetOpExecCacheManager()
{
    OP_LOGI("Get op exec cache manager.");
    g_opExecCacheManager.IncreaseUseCount();
    return PtrCastTo<void>(&g_opExecCacheManager);
}

void ReleaseOpExecCacheManager(void* ptr)
{
    if (ptr == nullptr) {
        OP_LOGI("Release op exec cache manager, but get nullptr.");
        return;
    }
    OP_LOGI("Release op exec cache manager.");
    OpExecCacheManager* manager = PtrCastTo<OpExecCacheManager>(ptr);
    manager->DecreaseUseCount();
}

void DisableOpCacheCount()
{
    OP_LOGI("Disable op cache count.");
    g_enableOpCacheCount.store(false);
}

void ReinitOpCacheManager()
{
    g_opExecCacheManager.ClearCacheManually();
    g_opExecCacheManager.InitCache();
}

OpExecCacheWrap* CreateCacheWrap(OpExecCache* opExecCache) { return new OpExecCacheWrap(opExecCache); }

OpCacheContext& GetOpCacheContext()
{
    static thread_local OpCacheContext ctx;
    return ctx;
}

OpCacheGuard::~OpCacheGuard()
{
    OP_LOGI("OpCacheGuard destruct");
    DisableOpCacheCount();
}

void OpExecCache::OldCacheClear()
{
    if (shrinkList_.Empty()) {
        return;
    }
    ListHead* curNode = &shrinkList_;
    ListHead* nextNode = curNode->next_;
    while (nextNode != &shrinkList_) {
        curNode = nextNode;
        nextNode = curNode->next_;
        g_opExecCacheManager.SubmitGcTask(curNode);
    }
    shrinkList_.next_ = &shrinkList_;
}

bool GetFromCache(aclOpExecutor** executor, uint64_t* workspaceSize)
{
    OpCacheKey key;
    SetOpCacheKey(key);
    auto cache = GetOpExecCache(key);
    if (cache != nullptr) {
        OpExecCacheWrap* cacheWrap = CreateCacheWrap(cache);
        *executor = reinterpret_cast<aclOpExecutor*>(cacheWrap);
        *workspaceSize = cache->GetWorkspaceSize();
        return true;
    }
    return false;
}

OpCacheKey::OpCacheKey(const OpCacheKey& key)
{
    buf = key.buf;
    len = key.len;
}

ge::AscendString OpCacheKey::ToString()
{
    std::stringstream ss;
    ss << "0x";
    for (size_t i = 0U; i < len; ++i) {
        ss << std::setfill('0') << std::setw(OP_BYTE_FOR_HEX) << std::hex << +buf[i];
    }
    return ge::AscendString((ss.str()).c_str());
}

bool operator==(const OpCacheKey& lhs, const OpCacheKey& rhs)
{
    auto& opCacheValue = static_cast<const OpCacheValue&>(rhs);
    OpExecCache* cache = opCacheValue.cache_;
    if (cache == nullptr) {
        return false;
    }
    OpCacheKey key = cache->GetOpCacheKey();
    // 防御：任一侧 buf 为空或 len 为 0 视为无效，避免 memcmp 解引用空指针
    if (lhs.buf == nullptr || lhs.len == 0 || key.buf == nullptr || key.len == 0) {
        OP_LOGW("op cache key is invalid, lhs buf is null: %d, lhs len: %zu, key buf is null: %d, key len: %zu",
                lhs.buf == nullptr, lhs.len, key.buf == nullptr, key.len);
        return false;
    }
    return lhs.len == key.len && (memcmp(PtrCastTo<char>(lhs.buf), PtrCastTo<char>(key.buf), lhs.len) == 0);
}

OpExecCache::OpExecCache()
{
    InitCacheData();
    opExecCacheDfx_ = new OpExecCacheDfx();
}

OpExecCache::~OpExecCache()
{
    OP_LOGD("destruct OpExecCache %p", this);
    OldCacheClear();
    if (hasExclusiveMem_) {
        delete[] static_cast<char*>(cacheBuf_);
    }
    for (auto it : cacheTensorInfoLists_) {
        op::internal::DestoryTensorsCached(it);
    }
    if (key_.buf) {
        delete[] key_.buf;
    }
    delete opExecCacheDfx_;
}

void OpExecCache::InitOpCacheKey()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (tlsData->threadLocalContext.cacheHashKeyLen_ != 0) {
        key_.len = tlsData->threadLocalContext.cacheHashKeyLen_;
        key_.buf = new uint8_t[key_.len];
        OP_LOGI("key_.len is %zu", key_.len);
        OP_CHECK(memcpy_s(key_.buf, key_.len, tlsData->threadLocalContext.cacheHashKey_, key_.len) == EOK,
                 OP_LOGW("Failed to memcpy in op cache."),
                 ;);
    }
}

void OpExecCache::InitCacheData()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    auto& tlsCachedStorageList = tlsData->threadLocalContext.cachedStorageList_;
    auto& tlsCachedStorageListSize = tlsData->threadLocalContext.cachedStorageListSize_;
    cachedStorageList_.assign(tlsCachedStorageList.begin(), tlsCachedStorageList.begin() + tlsCachedStorageListSize);
    hashKey_ = tlsData->threadLocalContext.hashKey_;
    InitOpCacheKey();
    if (tlsData->threadLocalContext.usePTAHash_) {
        l2Name_ = tlsData->threadLocalContext.cacheApi_;
    } else {
        l2Name_ = tlsData->threadLocalContext.logInfo_.l2ApiName;
    }
}

aclnnStatus OpExecCache::RecordAddrRule(const aclTensor* t, AddrRule& rule)
{
    if (t->IsFromWorkspace()) {
        rule.isWorkspace = true;
        rule.workspaceOffset = t->GetWorkspaceOffset();
        return ACLNN_SUCCESS;
    }
    auto s = t->GetStorage();
    // l0 use l2 tensor
    auto it = std::find(cachedStorageList_.begin(), cachedStorageList_.end(), s);
    if (it != cachedStorageList_.end()) {
        auto idx = std::distance(cachedStorageList_.begin(), it);
        rule.isWorkspace = false;
        rule.l2TensorInx = idx;
        rule.l2TensorOffset = op::CalcShapeBytes(t->GetViewOffset() + t->GetStorageOffset(), t->GetDataType(), true);
        return ACLNN_SUCCESS;
    }

    if (storageRelation_.find(s) == storageRelation_.end()) {
        OP_LOGW("record addr rule find tensor relation fail");
        return ACLNN_ERR_INNER;
    }
    // viewCopy or createView product l0 tensor and l2 tensor relation
    auto rel = storageRelation_[s];
    it = std::find(cachedStorageList_.begin(), cachedStorageList_.end(), rel);
    if (it == cachedStorageList_.end()) {
        OP_LOGW("record addr rule find tensor fail");
        return ACLNN_ERR_INNER;
    }
    auto idx = std::distance(cachedStorageList_.begin(), it);
    rule.isWorkspace = false;
    rule.l2TensorInx = idx;
    rule.l2TensorOffset = op::CalcShapeBytes(t->GetViewOffset() + t->GetStorageOffset(), t->GetDataType(), true);
    return ACLNN_SUCCESS;
}

void* OpExecCache::AddLaunchTensor(const aclTensor* t, size_t dataLen)
{
    if (t->IsFromWorkspace()) {
        addrUpdateRelation_[cacheOffset_] = AddrRule(true, t->GetWorkspaceOffset(), 0, 0);
        return AddLaunchData(dataLen);
    }
    auto s = t->GetStorage();
    // l0 use l2 tensor
    auto it = std::find(cachedStorageList_.begin(), cachedStorageList_.end(), s);
    if (it != cachedStorageList_.end()) {
        auto idx = std::distance(cachedStorageList_.begin(), it);
        addrUpdateRelation_[cacheOffset_] = AddrRule(
            false, 0, idx, op::CalcShapeBytes(t->GetViewOffset() + t->GetStorageOffset(), t->GetDataType(), true));
        return AddLaunchData(dataLen);
    }

    if (storageRelation_.find(s) == storageRelation_.end()) {
        OP_LOGW("op executor cache find tensor relation fail");
        MarkOpCacheInvalid();
        return nullptr;
    }
    // viewCopy or createView product l0 tensor and l2 tensor relation
    auto rel = storageRelation_[s];
    it = std::find(cachedStorageList_.begin(), cachedStorageList_.end(), rel);
    if (it == cachedStorageList_.end()) {
        OP_LOGW("op executor cache find tensor fail");
        MarkOpCacheInvalid();
        return nullptr;
    }
    auto idx = std::distance(cachedStorageList_.begin(), it);
    addrUpdateRelation_[cacheOffset_] = AddrRule(
        false, 0, idx, op::CalcShapeBytes(t->GetViewOffset() + t->GetStorageOffset(), t->GetDataType(), true));
    return AddLaunchData(dataLen);
}

void* OpExecCache::AddLaunchData(size_t dataLen)
{
    // if cache buf not enough, abandon cache.
    if (cacheOffset_ + dataLen > cacheCap_) {
        MarkOpCacheInvalid();
        OP_LOGW("cache buf has full, abandon current cache.");
        if (dataLen > cacheCap_) {
            OP_LOGW("cache buf cap is not enough for once cache data!");
        }
        return nullptr;
    }
    void* buf = op::internal::PtrShift(cacheBuf_, cacheOffset_);
    cacheOffset_ += dataLen;
    return buf;
}

void OpExecCache::SetCacheBuf(void* buf) { cacheBuf_ = buf; }

void OpExecCache::UpdateTensorAddr(void* workspaceAddr, const std::vector<void*>& tensors)
{
    for (auto it = addrUpdateRelation_.begin(); it != addrUpdateRelation_.end(); ++it) {
        void* buf = op::internal::PtrShift(cacheBuf_, it->first);
        void* newAddr;
        if (it->second.isWorkspace) {
            newAddr = op::internal::PtrShift(workspaceAddr, it->second.workspaceOffset);
        } else {
            newAddr = op::internal::PtrShift(tensors[it->second.l2TensorInx], it->second.l2TensorOffset);
        }
        void** p = op::internal::PtrCastTo<void*>(buf);
        *p = newAddr;
    }
}

void OpExecCache::SetBlockDim(uint32_t blockDim) { numBlocks_.push_back(blockDim); }

void OpExecCache::SetNumBlocks(uint32_t numBlocks) { numBlocks_.push_back(numBlocks); }

void OpExecCache::SetCacheTensorInfo(void* infoLists) { cacheTensorInfoLists_.push_back(infoLists); }

void* OpExecCache::GetCacheTensorInfo(int index) { return cacheTensorInfoLists_[index]; }

void OpExecCache::AddTensorRelation(const aclTensor* tensorOut, const aclTensor* tensorMiddle)
{
    if (IsOpCacheValid()) {
        auto storageOut = tensorOut->GetStorage();
        auto storageMiddle = tensorMiddle->GetStorage();
        storageRelation_[storageMiddle] = storageOut;
    }
}

std::unordered_map<const aclStorage*, const aclStorage*> OpExecCache::GetStorageRelation() { return storageRelation_; }

void OpExecCache::NewLaunchCache(size_t* offset, size_t* cap, R runner)
{
    taskQueue_.push_back({cacheOffset_, runner});
    *offset = cacheOffset_;
    *cap = cacheCap_ - cacheOffset_;
}

void OpExecCache::Finalize()
{
    OldCacheClear();
    if (!IsOpCacheValid()) {
        return;
    }
    // Without going through phase two, cache can't be used
    if (cacheBuf_ == nullptr) {
        MarkOpCacheInvalid();
        return;
    }
    size_t cacheSize = K_CACHE_SIZE_ALIGN;
    if (cacheOffset_ > 0) {
        cacheSize = op::internal::AlignSize(cacheOffset_, K_CACHE_SIZE_ALIGN);
    }
    auto newCacheBuf = new char[cacheSize];
    OP_CHECK(memcpy_s(static_cast<void*>(newCacheBuf), cacheSize, cacheBuf_, cacheSize) == EOK,
             OP_LOGW("Failed to memcpy in op cache."),
             ;);
    cacheBuf_ = static_cast<void*>(newCacheBuf);
    hasExclusiveMem_ = true;
    cachedStorageList_.clear();
    storageRelation_.clear();
}

void OpExecCache::SetWorkspaceSize(uint64_t val) { workspaceSize_ = val; }

uint64_t OpExecCache::GetWorkspaceSize() const { return workspaceSize_; }

void OpExecCacheDfx::SetTaskInfo(const TaskInfo& taskInfo) { taskInfo_.push_back(taskInfo); }

uint64_t OpExecCache::GetHash() const { return hashKey_; }

OpCacheKey OpExecCache::GetOpCacheKey() const { return key_; }

void OpExecCache::MarkOpCacheInvalid()
{
    OP_CHECK(!(CanUse()), OP_LOGI("OpExecCache has been can used, cant change to invalid."), return);
    hashKey_ = 0;
    // 不释放/置空 buf 仅置 len 为 0，避免 operator== 跨线程读到 buf 为空而 len 非 0 的中间态导致崩溃
    key_.len = 0;
    OP_LOGI("key_.len %zu", key_.len);
}

bool OpExecCache::IsOpCacheValid()
{
    OP_LOGI("hash Key: %lu, key_.len %zu", hashKey_, key_.len);
    return hashKey_ || (key_.buf && key_.len);
}

aclnnStatus OpExecCache::DoSummaryProfiling(int index)
{
    if (opExecCacheDfx_->GetTaskType(index) == MSPROF_GE_TASK_TYPE_MIX_AIV ||
        opExecCacheDfx_->GetTaskType(index) == MSPROF_GE_TASK_TYPE_MIX_AIC) {
        op::internal::ReportNodeContextIdInfo(opExecCacheDfx_->GetProfilingInfoId(index).summaryItemId_);
    }
    // only level2 profiling need to report addition info
    if (!op::internal::opProfilingSwitch.additionInfoFlag) {
        return ACLNN_SUCCESS;
    }
    void* tensorInfoLists = cacheTensorInfoLists_[index];
    return DoReportAdditionInfo(tensorInfoLists, opExecCacheDfx_->GetTaskInfo(index),
                                opExecCacheDfx_->GetProfilingInfoId(index));
}

void OpExecCache::RestoreThreadLocal(int index)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.numBlocks_ = numBlocks_[index];
    tlsData->threadLocalContext.profilingInfoId_ = opExecCacheDfx_->GetProfilingInfoId(index);
}

aclnnStatus OpExecCache::Run(void* workspaceAddr, const aclrtStream stream, const std::vector<void*>& tensors)
{
    int index = 0;
    UpdateTensorAddr(workspaceAddr, tensors);
    for (Task& t : taskQueue_) {
        RestoreThreadLocal(index);
        {
            OpDfxGuard kernelLaunchGuard(opExecCacheDfx_->GetProfilingInfoId(index).summaryItemId_,
                                         DfxProfilingType::DfxProfilingKernelLaunch);
            aclnnStatus result = std::get<1>(t)(stream, op::internal::PtrShift(cacheBuf_, std::get<0>(t)));
            OP_CHECK(result == ACLNN_SUCCESS, OP_LOGE(result, "OpExecCache run fail."), return result);
        }
        if (IsExceptionDumpEnable()) {
            DoExceptionDump(cacheTensorInfoLists_[index], workspaceAddr, tensors,
                            opExecCacheDfx_->GetExceptionDumpInfo(index), stream);
        }
        if (op::internal::opProfilingSwitch.kernelLaunchFlag) {
            DoSummaryProfiling(index);
        }
        if (GetThreadLocalContext().cacheOpInfoSwitch_) {
            void* tensorInfoLists = cacheTensorInfoLists_[index];
            TaskInfo taskInfo = opExecCacheDfx_->GetTaskInfo(index);
            ReportCacheOpInfoFromCache(taskInfo, tensorInfoLists, g_opCacheTlsData.threadLocalContext.numBlocks_,
                                       opExecCacheDfx_->GetProfilingInfoId(index));
        }
        index++;
    }
    return ACLNN_SUCCESS;
}

bool OpExecCache::CanUse()
{
    const char* current;
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    if (tlsData->threadLocalContext.usePTAHash_) {
        current = tlsData->threadLocalContext.cacheApi_;
    } else {
        current = tlsData->threadLocalContext.logInfo_.l2ApiName;
    }
    if (current != l2Name_) {
        OP_LOGW("can not hit cache, may be hash conflict, current: %s, cache: %s", current, l2Name_);
        return false;
    }
    return canUse_.load();
}

void OpExecCache::SetUse()
{
    if (!IsOpCacheValid()) {
        return;
    }
    canUse_.store(true);
}

OpCacheValue::~OpCacheValue()
{
    if (cache_) {
        delete cache_;
        cache_ = nullptr;
    }
}

OpCacheValue::OpCacheValue(const OpCacheValue& value) : OpCacheKey(value), cache_(value.cache_)
{
    ListHead::next_ = this;
    ListHead::prev_ = this;
    HlistNode::next_ = nullptr;
    HlistNode::pprev_ = nullptr;
}

OpCacheValue::OpCacheValue(OpCacheValue&& value) : OpCacheKey(value), cache_(value.cache_)
{
    ListHead::next_ = this;
    ListHead::prev_ = this;
    HlistNode::next_ = nullptr;
    HlistNode::pprev_ = nullptr;

    value.cache_ = nullptr;
}

OpCacheValue& OpCacheValue::operator=(const OpCacheValue& value)
{
    cache_ = value.cache_;
    OpCacheKey::buf = value.OpCacheKey::buf;
    OpCacheKey::len = value.OpCacheKey::len;
    return *this;
}

OpCacheValue& OpCacheValue::operator=(OpCacheValue&& value)
{
    cache_ = value.cache_;
    OpCacheKey::buf = value.OpCacheKey::buf;
    OpCacheKey::len = value.OpCacheKey::len;
    value.cache_ = nullptr;
    return *this;
}

OpExecCacheManager::OpExecCacheManager() : cacheLimit_(GetCacheSizeLimit()) { cache2_.init(cacheLimit_); }

OpExecCacheManager::~OpExecCacheManager()
{
    WaitCacheCompleteUse();
    DeleteCache1();
    OP_LOGI("delete op exec cache manager");
    if (gcInitialize_ && consumer_.joinable()) {
        threadStop_.store(true);
        consumer_.join();
    }
}

void OpExecCacheManager::InitCache() { cache2_.init(cacheLimit_); }

void OpExecCacheManager::ClearCacheManually()
{
    WaitCacheCompleteUse();
    DeleteCache1();
    cache_.clear();
    cache2_.clear();
    op::internal::GetThreadLocalContext().cacheHasFull_ = false;
}

void OpExecCacheManager::WaitCacheCompleteUse()
{
    OP_LOGI("there are %ld cache in use now", useCount_.load());
    if (!g_enableOpCacheCount.load()) {
        OP_LOGI("OpCache count is disabled, no need to wait cache complete use");
        return;
    }
    int64_t waitTime = 0;
    while (true) {
        waitTime++;
        if (useCount_.load() <= 0 || waitTime >= CACHE_DESTRUCT_MAX_WAIT_TIME) {
            break;
        }
        usleep(SLEEP_ONE_SECOND);
    }
}

void OpExecCacheManager::DeleteCache1()
{
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->second != nullptr) {
            delete it->second;
        }
    }
}

OpExecCache* OpExecCacheManager::GetOpExecCache(uint64_t hash)
{
    if (g_opCacheTlsData.threadLocalContext.opConfigInfo_.usePcieAddr) {
        OP_LOGI("Skip cache get, current op has tensor using PCIe addr");
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(lock_);
    auto it = cache_.find(hash);
    if (it == cache_.end()) {
        return nullptr;
    }
    auto cache = it->second;
    if (!cache->CanUse()) {
        return nullptr;
    }
    return cache;
}

OpExecCache* OpExecCacheManager::GetOpExecCache(OpCacheKey& key)
{
    if (g_opCacheTlsData.threadLocalContext.opConfigInfo_.usePcieAddr) {
        OP_LOGI("Skip cache get, current op has tensor using PCIe addr");
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(lock_);
    auto it = cache2_.find(key);
    if (it == cache2_.end()) {
        return nullptr;
    }
    OpCacheValue& value = *it;
    if (!value.cache_->CanUse()) {
        return nullptr;
    }
    OP_LOGD("Get op cache key %s value %p", value.ToString().GetString(), value.cache_);
    return value.cache_;
}

size_t OpExecCacheManager::GetCacheSizeLimit()
{
    const char_t* cacheLimit = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ACLNN_CACHE_LIMIT, cacheLimit);
#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
    size_t c = 1;
#else
    size_t c = K_CACHE_LIMIT;
#endif
    if (cacheLimit) {
        try {
            c = std::stoull(cacheLimit);
        } catch (const std::exception& e) {
            OP_LOGD("Env variable ACLNN_CACHE_LIMIT[%s] is invalid! must be a number!", cacheLimit);
        }
    }
    c = std::min(K_MAX_CACHE_LIMIT, c);
    OP_LOGI("cachelimit is %zu", c);
    return c;
}

bool OpExecCacheManager::AddOpExecCache(OpExecCache* exec)
{
    if (g_opCacheTlsData.threadLocalContext.opConfigInfo_.usePcieAddr) {
        OP_LOGI("Skip cache add, current op has tensor using PCIe addr");
        delete exec;
        return false;
    }
    if (!exec->IsOpCacheValid()) {
        delete exec;
        return false;
    }
    {
        bool ret = false;
        std::lock_guard<std::mutex> guard(lock_);
        if (cache_.size() >= cacheLimit_) {
            OP_LOGW("op cache is full");
            op::internal::GetThreadLocalContext().cacheHasFull_ = true;
            delete exec;
            return false;
        }
        uint64_t hash = exec->GetHash();
        if (hash && cache_.find(hash) == cache_.end()) {
            cache_[hash] = exec;
            ret = true;
        }

        if (cache2_.size() >= cacheLimit_) {
            OP_LOGW("op cache is full");
            ShrinkCache(K_CACHE_SHRINK_NUM, exec->GetShrinkList());
        }
        OpCacheKey key = exec->GetOpCacheKey();
        if (key.buf && key.len && cache2_.find(key) == cache2_.end()) {
            OpCacheValue value(exec, key);
            cache2_[key] = std::move(value);
            ret = true;
            OP_LOGD("Add op cache key %s value %p", key.ToString().GetString(), exec);
        }
        if (ret) {
            return ret;
        }
    }
    delete exec;
    return false;
}

void OpExecCacheManager::RemoveOpExecCache(OpExecCache* exec)
{
    if (exec == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> guard(lock_);
        uint64_t hash = exec->GetHash();
        if (hash && cache_.find(hash) != cache_.end()) {
            cache_.erase(cache_.find(hash));
            return;
        }
        OpCacheKey key = exec->GetOpCacheKey();
        if (key.buf && key.len && cache2_.find(key) != cache2_.end()) {
            OpCacheValue* cacheVal = cache2_.find(key).operator->();
            cache2_.erase(*cacheVal);
            cacheVal->cache_ = nullptr;
            delete cacheVal;
            return;
        }
    }
}

void OpExecCacheManager::ShrinkCache(const size_t num, ListHead* shrinkList)
{
    OP_LOGI("op cache size %zu before shrink", cache2_.size());
    for (size_t i = 0; i < num; i++) {
        if (cache2_.rbegin() != cache2_.rend()) {
            OpCacheValue* value = cache2_.rbegin().operator->();
            cache2_.erase(*value);
            value->ListHead::Add(shrinkList);
            if (value->cache_ != nullptr && value->cache_->GetOpCacheKey().buf != nullptr) {
                OP_LOGD("Delete op cache key %s value %p", value->ToString().GetString(), value->cache_);
            } else {
                OP_LOGD("Delete invalid cache value %p", value->cache_);
            }
        } else {
            break;
        }
    }
    OP_LOGI("op cache size %zu after shrink", cache2_.size());
}

void OpExecCacheManager::Start()
{
    auto f = [this]() {
        OP_LOGI("start op cache gc thread");
        ListHead* task = nullptr;
        while (true) {
            this->gcQueue_.Dequeue(task);
            if (task != nullptr) {
                OpCacheValue* value = reinterpret_cast<OpCacheValue*>(task);
                delete value;
                task = nullptr;
                continue;
            }
            if (this->threadStop_.load()) {
                OP_LOGI("stop op cache gc thread");
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };
    std::thread consumer(f);
    consumer_ = std::move(consumer);
};

void OpExecCacheManager::SubmitGcTask(ListHead* c)
{
    if (!gcInitialize_) {
        gcInitialize_ = true;
        Start();
    }
    if (!gcQueue_.Enqueue(c)) {
        delete c;
    }
}

OpExecCacheWrap::OpExecCacheWrap(OpExecCache* cache) : opExecCache_(cache)
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    auto& tlsCachedTensorList = tlsData->threadLocalContext.cachedTensorList_;
    auto& tlsCachedTensorListSize = tlsData->threadLocalContext.cachedTensorListSize_;
    cachedTensorList_.assign(tlsCachedTensorList.begin(), tlsCachedTensorList.begin() + tlsCachedTensorListSize);
    opLogInfo_ = tlsData->threadLocalContext.logInfo_;
    hugeMemPoolIndex_ = tlsData->threadLocalContext.poolIndex_;
    opExecCacheManager_ = GetOpExecCacheManager();
    OP_LOGI("Op exec cache get device ptr list: %s. Hugemem trace: huge mem pool index: %d", ReportAddr().c_str(),
            hugeMemPoolIndex_);
}

OpExecCacheWrap::~OpExecCacheWrap()
{
    OpCacheThreadLocalData* tlsData = &g_opCacheTlsData;
    tlsData->threadLocalContext.Init();
    tlsData->threadLocalContext.poolIndex_ = hugeMemPoolIndex_;
    ReleaseOpExecCacheManager(opExecCacheManager_);
    OP_LOGI("Hugemem trace: huge mem pool index: %d", hugeMemPoolIndex_);
}

aclnnStatus OpExecCacheWrap::Run(void* workspaceAddr, const aclrtStream stream)
{
    return opExecCache_->Run(workspaceAddr, stream, cachedTensorList_);
}

std::string OpExecCacheWrap::ReportAddr()
{
    std::ostringstream oss;
    for (void* ptr : cachedTensorList_) {
        oss << ptr;
        oss << ", ";
    }
    return oss.str();
}

void OpCacheContext::SetOpCache(OpExecCache* cache) { cache_ = cache; }

OpExecCache* OpCacheContext::GetOpCache() { return cache_; }

static const char* GetEnvOfDisableL2Cache()
{
    const char* value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_DISABLE_L2_CACHE, value);
    return value;
}

bool CheckCacheable()
{
    bool cacheDisable = (op::internal::IsDumpEnable() || op::internal::IsExceptionDumpEnable() ||
                         op::internal::IsOverflowDumpEnable() ||
                         (op::internal::GetOpProfilingRecordArgFlag() ||
                          internal::opProfilingSwitch.level2ProfilingFlag));
    if (cacheDisable) {
        g_opCacheTlsData.threadLocalContext.cacheHasFull_ = true;
        return false;
    }
    static const char* pathVar = GetEnvOfDisableL2Cache();
    if (pathVar != nullptr) {
        g_opCacheTlsData.threadLocalContext.cacheHasFull_ = true;
        return false;
    }
    return true;
}
} // namespace internal
} // namespace op