/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_ctx_def.h"
#include <cstdlib>
#include <securec.h>
#include "opdev/op_log.h"

namespace op::internal {
// 扩容倍数因子
constexpr size_t BUFFER_EXPANSION_FACTOR = 2;
// 扩容溢出阈值：超过此值再乘以 BUFFER_EXPANSION_FACTOR 会溢出
constexpr size_t MAX_CAPACITY_BEFORE_OVERFLOW = SIZE_MAX / BUFFER_EXPANSION_FACTOR;

ExpandableRtsArgBuffer::~ExpandableRtsArgBuffer()
{
    if (baseAddr_) {
        std::free(baseAddr_);
        baseAddr_ = nullptr;
    }
}

aclnnStatus ExpandableRtsArgBuffer::Init(size_t launchArgCap, size_t tilingHostDataCap)
{
    if (baseAddr_ != nullptr) {
        OP_LOGW("Init called when buffer already initialized, current size %zu, "
                "requested launch_arg capacity %zu, tiling_host_data capacity %zu.",
                totalSize_, launchArgCap, tilingHostDataCap);
        return ACLNN_SUCCESS;
    }
    totalSize_ = sizeof(TilingData) + launchArgCap + tilingHostDataCap;
    baseAddr_ = std::malloc(totalSize_);
    OP_CHECK(baseAddr_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc size %zu.", totalSize_),
             return ACLNN_ERR_INNER);
    launchArgCapacity_ = launchArgCap;
    launchArgSize_ = 0;
    tilingHostDataStart_ = sizeof(TilingData) + launchArgCapacity_;
    tilingHostDataCapacity_ = tilingHostDataCap;
    tilingHostDataSize_ = 0;
    OP_LOGI("ExpandableRtsArgBuffer init: baseAddr=%p, totalSize=%zu, "
            "launchArgCapacity=%zu, tilingHostDataCapacity=%zu",
            baseAddr_, totalSize_, launchArgCapacity_, tilingHostDataCapacity_);
    return ACLNN_SUCCESS;
}

aclnnStatus ExpandableRtsArgBuffer::EnsureLaunchArgCapacity(size_t requiredSize)
{
    if (requiredSize <= launchArgCapacity_) {
        return ACLNN_SUCCESS;
    }
    return ExpandLaunchArg(requiredSize);
}

aclnnStatus ExpandableRtsArgBuffer::UpdateTilingDataSize(size_t tilingDataSize)
{
    if (tilingHostDataSize_ >= tilingDataSize) {
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Current tiling host data size %zu, update tiling data size to %zu, need to seek %zu.", tilingHostDataSize_,
            tilingDataSize, tilingDataSize - tilingHostDataSize_);
    alignedTilingDataSize_ = tilingDataSize;
    return SeekTilingHostData(tilingDataSize - tilingHostDataSize_);
}

aclnnStatus ExpandableRtsArgBuffer::AppendTilingHostData(const void* data, size_t len)
{
    size_t oldSize = tilingHostDataSize_;
    OP_CHECK(SeekTilingHostData(len) == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER, "failed to seek tiling host data, len %zu.", len), return ACLNN_ERR_INNER);
    void* writePos = static_cast<uint8_t*>(GetTilingDataAddr()) + oldSize;
    size_t availableSize = tilingHostDataCapacity_ - oldSize;
    OP_CHECK(memcpy_s(writePos, availableSize, data, len) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy tiling host data."), return ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

aclnnStatus ExpandableRtsArgBuffer::SeekTilingHostData(size_t len)
{
    size_t oldSize = tilingHostDataSize_;
    if (tilingHostDataSize_ + len > tilingHostDataCapacity_) {
        size_t requiredCapacity = tilingHostDataSize_ + len;
        OP_CHECK(
            ExpandTilingHostData(requiredCapacity) == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER, "failed to expand tiling host data, required capacity %zu.", requiredCapacity),
            return ACLNN_ERR_INNER);
    }
    tilingHostDataSize_ += len;
    OP_LOGD("seek tiling host data len %zu, tiling host data used size from %zu to %zu.", len, oldSize,
            tilingHostDataSize_);
    return ACLNN_SUCCESS;
}

aclnnStatus ExpandableRtsArgBuffer::ExpandLaunchArg(size_t requiredCapacity)
{
    OP_CHECK(launchArgCapacity_ != 0, OP_LOGE(ACLNN_ERR_INNER, "launchArgCapacity_ is 0, Init not called."),
             return ACLNN_ERR_INNER);

    size_t oldLaunchCap = launchArgCapacity_;
    size_t newLaunchCap = oldLaunchCap;
    while (newLaunchCap < requiredCapacity) {
        // 检查乘法溢出，防止死循环
        OP_CHECK(newLaunchCap <= MAX_CAPACITY_BEFORE_OVERFLOW,
                 OP_LOGE(ACLNN_ERR_INNER, "launch arg capacity overflow, current %zu, required %zu.", newLaunchCap,
                         requiredCapacity),
                 return ACLNN_ERR_INNER);
        newLaunchCap *= BUFFER_EXPANSION_FACTOR;
    }

    size_t oldTilingHostDataStart = tilingHostDataStart_;
    size_t newTotalSize = sizeof(TilingData) + newLaunchCap + tilingHostDataCapacity_;

    OP_LOGW("Expand launch_arg capacity from %zu to %zu, causing buffer total size to grow from %zu to %zu",
            oldLaunchCap, newLaunchCap, totalSize_, newTotalSize);

    void* newAddr = std::malloc(newTotalSize);
    OP_CHECK(newAddr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc size %zu.", newTotalSize),
             return ACLNN_ERR_INNER);

    // Step 1: 拷贝 TilingData 头部
    if (memcpy_s(newAddr, sizeof(TilingData), baseAddr_, sizeof(TilingData)) != EOK) {
        std::free(newAddr);
        OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy TilingData header.");
        return ACLNN_ERR_INNER;
    }

    // Step 2: 跳过 launch_arg — 不拷贝，此时没有有效数据

    // Step 3: 拷贝 tiling_host_data 已用部分
    size_t newTilingHostDataStart = sizeof(TilingData) + newLaunchCap;
    if (tilingHostDataSize_ > 0) {
        void* src = static_cast<uint8_t*>(baseAddr_) + oldTilingHostDataStart;
        void* dst = static_cast<uint8_t*>(newAddr) + newTilingHostDataStart;
        if (memcpy_s(dst, tilingHostDataCapacity_, src, tilingHostDataSize_) != EOK) {
            std::free(newAddr);
            OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy tiling_host_data.");
            return ACLNN_ERR_INNER;
        }
    }

    // Step 4: 更新状态
    std::free(baseAddr_);
    baseAddr_ = newAddr;
    launchArgCapacity_ = newLaunchCap;
    tilingHostDataStart_ = newTilingHostDataStart;
    totalSize_ = newTotalSize;

    // Step 5: 递增版本计数器，刷新注册指针
    generation_++;
    UpdateRegisteredPtrs();
    return ACLNN_SUCCESS;
}

aclnnStatus ExpandableRtsArgBuffer::ExpandTilingHostData(size_t requiredCapacity)
{
    OP_CHECK(tilingHostDataCapacity_ != 0, OP_LOGE(ACLNN_ERR_INNER, "tilingHostDataCapacity_ is 0, Init not called."),
             return ACLNN_ERR_INNER);

    size_t oldTilingHostDataCap = tilingHostDataCapacity_;
    size_t newTilingHostDataCap = oldTilingHostDataCap;
    while (newTilingHostDataCap < requiredCapacity) {
        // 检查乘法溢出，防止死循环
        OP_CHECK(newTilingHostDataCap <= MAX_CAPACITY_BEFORE_OVERFLOW,
                 OP_LOGE(ACLNN_ERR_INNER, "tiling host data capacity overflow, current %zu, required %zu.",
                         newTilingHostDataCap, requiredCapacity),
                 return ACLNN_ERR_INNER);
        newTilingHostDataCap *= BUFFER_EXPANSION_FACTOR;
    }

    size_t newTotalSize = sizeof(TilingData) + launchArgCapacity_ + newTilingHostDataCap;

    OP_LOGW("Expand tiling_host_data capacity from %zu to %zu, causing buffer total size to grow from %zu to %zu",
            oldTilingHostDataCap, newTilingHostDataCap, totalSize_, newTotalSize);

    void* newAddr = std::malloc(newTotalSize);
    OP_CHECK(newAddr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc size %zu.", newTotalSize),
             return ACLNN_ERR_INNER);

    // Step 1: 拷贝 TilingData 头部
    if (memcpy_s(newAddr, sizeof(TilingData), baseAddr_, sizeof(TilingData)) != EOK) {
        std::free(newAddr);
        OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy TilingData header.");
        return ACLNN_ERR_INNER;
    }

    // Step 2: 拷贝 launch_arg 已用部分：launch_arg 在区域后端，仅拷贝后端 launchArgSize_ 字节
    if (launchArgSize_ > 0) {
        size_t launchArgUsedStart = sizeof(TilingData) + launchArgCapacity_ - launchArgSize_;
        if (memcpy_s(static_cast<uint8_t*>(newAddr) + launchArgUsedStart, launchArgSize_,
                     static_cast<uint8_t*>(baseAddr_) + launchArgUsedStart, launchArgSize_) != EOK) {
            std::free(newAddr);
            OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy launch_arg used portion.");
            return ACLNN_ERR_INNER;
        }
    }

    // Step 3: 拷贝 tiling_host_data 已用部分
    if (tilingHostDataSize_ > 0) {
        void* src = static_cast<uint8_t*>(baseAddr_) + tilingHostDataStart_;
        void* dst = static_cast<uint8_t*>(newAddr) + tilingHostDataStart_;
        if (memcpy_s(dst, newTilingHostDataCap, src, tilingHostDataSize_) != EOK) {
            std::free(newAddr);
            OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy tiling_host_data.");
            return ACLNN_ERR_INNER;
        }
    }

    // Step 4: 更新状态
    std::free(baseAddr_);
    baseAddr_ = newAddr;
    tilingHostDataCapacity_ = newTilingHostDataCap;
    totalSize_ = newTotalSize;

    // Step 5: 递增版本计数器，刷新注册指针
    generation_++;
    UpdateRegisteredPtrs();
    return ACLNN_SUCCESS;
}

void ExpandableRtsArgBuffer::UpdateRegisteredPtrs()
{
    TilingData* tilingData = static_cast<TilingData*>(baseAddr_);
    tilingData->data_ = GetTilingDataAddr();
    tilingData->capacity_ = tilingHostDataCapacity_;

    OP_LOGD("UpdateRegisteredPtrs: tilingData %p, data_ %p, capacity_ %zu, "
            "holderTilingDataPtr_ %s, outputTilingDataPtr_ %s",
            tilingData, tilingData->data_, tilingData->capacity_, holderTilingDataPtr_ ? "valid" : "null",
            outputTilingDataPtr_ ? "valid" : "null");

    if (holderTilingDataPtr_) {
        *holderTilingDataPtr_ = tilingData;
    }
    if (outputTilingDataPtr_) {
        *outputTilingDataPtr_ = tilingData;
    }
}

size_t ExpandableRtsArgBuffer::GetHostDataSize() const
{
    if (tilingHostDataSize_ < alignedTilingDataSize_) {
        OP_LOGW("Invalid state: tilingHostDataSize_=%zu < alignedTilingDataSize_=%zu", tilingHostDataSize_,
                alignedTilingDataSize_);
        return 0;
    }
    return tilingHostDataSize_ - alignedTilingDataSize_;
}

void ExpandableRtsArgBuffer::ResetTilingHostDataCursor()
{
    tilingHostDataSize_ = 0;
    alignedTilingDataSize_ = 0;
}

} // namespace op::internal
