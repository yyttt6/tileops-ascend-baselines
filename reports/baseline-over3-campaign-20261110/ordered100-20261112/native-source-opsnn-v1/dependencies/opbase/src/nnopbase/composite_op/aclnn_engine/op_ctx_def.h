/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_CTX_DEF_H__
#define __OP_CTX_DEF_H__
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "exe_graph/runtime/continuous_vector.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"

// Structs blow are from GE RT2.0. Must keep sync with GE.
namespace op::internal {
using AttrPtr = void*;
struct ComputeNodeInfo {
    const ge::char_t* node_type_;
    const ge::char_t* node_name_;
    size_t ir_inputs_num_;
    size_t inputs_num_;
    size_t outputs_num_;
    size_t ir_outputs_num_;
    size_t attr_size_;
    uint8_t reserved_[24]; // Reserved field, 32+8, do not directly use when only 8-byte left
    // following by input-AnchorInstanceInfo, inputs-outputs-CompileTimeTensorDesc, RuntimeAttrs,
    // output-AnchorInstanceInfo
    uint64_t place_holder;

    /*
    AnchorInstanceInfo[ir_inputs_num_]
    CompileTimeTensorDesc[inputs_num_ + outputs_num_]
    RuntimeAttrsDef
    */
};

struct AnchorInstanceInfo {
    uint32_t instance_start_;
    uint32_t instantiation_num_;
    uint8_t reserved_[40];
};

struct CompileTimeTensorDesc {
    DataType data_type_;
    gert::StorageFormat storage_format_;
    uint8_t reserved_[40];
};

struct RuntimeAttrsDef {
    size_t attr_num;
    uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
    size_t offset[0];
};

struct KernelExtendInfo {
    const ge::char_t* kernel_name_;
    const ge::char_t* kernel_type_;
};

struct TilingData;

struct ExpandableRtsArgBuffer {
public:
    ExpandableRtsArgBuffer() = default;
    ~ExpandableRtsArgBuffer();
    ExpandableRtsArgBuffer(const ExpandableRtsArgBuffer&) = delete;
    ExpandableRtsArgBuffer& operator=(const ExpandableRtsArgBuffer&) = delete;
    ExpandableRtsArgBuffer(ExpandableRtsArgBuffer&&) = delete;
    ExpandableRtsArgBuffer& operator=(ExpandableRtsArgBuffer&&) = delete;

    // 初始化：分别指定 launch_arg 和 tiling_host_data 的初始容量
    aclnnStatus Init(size_t launchArgCap, size_t tilingHostDataCap);

    // 扩容检测：使用版本计数器，每次扩容时递增。调用方通过比较前后版本号判断是否发生了扩容。
    size_t GetGeneration() const { return generation_; }

    // --- TilingData 头部 ---
    TilingData* GetTilingDataPtr() const { return static_cast<TilingData*>(baseAddr_); }

    // --- launch_arg 区域 ---
    size_t GetLaunchArgCapacity() const { return launchArgCapacity_; }
    size_t GetLaunchArgSize() const { return launchArgSize_; }
    void SetLaunchArgSize(size_t size) { launchArgSize_ = size; }
    aclnnStatus EnsureLaunchArgCapacity(size_t requiredSize);

    // --- tiling_host_data 区域 ---
    // 返回 tiling data 地址（也是 tiling_host_data 起始地址和 launch_arg 区域的结束位置）
    void* GetTilingDataAddr() const { return static_cast<uint8_t*>(baseAddr_) + tilingHostDataStart_; }
    void* GetTilingHostDataCurEndAddr() const
    {
        return static_cast<uint8_t*>(baseAddr_) + tilingHostDataStart_ + tilingHostDataSize_;
    }
    size_t GetTilingHostDataCapacity() const { return tilingHostDataCapacity_; }
    size_t GetTilingHostDataSize() const { return tilingHostDataSize_; }
    size_t GetAlignedTilingDataSize() const { return alignedTilingDataSize_; }
    size_t GetHostDataSize() const;
    aclnnStatus UpdateTilingDataSize(size_t tilingDataSize);
    aclnnStatus AppendTilingHostData(const void* data, size_t len);
    aclnnStatus SeekTilingHostData(size_t len);
    void ResetTilingHostDataCursor();

    // --- 注册需要刷新的指针 ---
    void RegisterHolderTilingDataPtr(TilingData** ptr) { holderTilingDataPtr_ = ptr; }
    void RegisterOutputTilingDataPtr(TilingData** ptr) { outputTilingDataPtr_ = ptr; }

private:
    aclnnStatus ExpandLaunchArg(size_t requiredCapacity);
    aclnnStatus ExpandTilingHostData(size_t requiredCapacity);
    void UpdateRegisteredPtrs();

    void* baseAddr_{nullptr};
    size_t totalSize_{0};

    // launch_arg 区域管理
    size_t launchArgCapacity_{0};
    size_t launchArgSize_{0};

    // tiling_host_data 区域管理
    size_t tilingHostDataStart_{0};
    size_t tilingHostDataCapacity_{0};
    size_t tilingHostDataSize_{0};
    size_t alignedTilingDataSize_{0}; // tiling 数据长度（对齐后）

    // 需要刷新的TilingData指针
    TilingData** holderTilingDataPtr_{nullptr};
    TilingData** outputTilingDataPtr_{nullptr};

    // 扩容版本计数器
    size_t generation_{0};
};

struct TilingData {
    size_t capacity_;
    size_t data_size_;
    void* data_;
    uint8_t reserved_[40];
};

// Tiling output
struct TilingCtxOutput {
    uint64_t* tilingKey_;
    int64_t* numBlocks_;
    bool* atomicCleanFlag_;
    TilingData* tilingData_;
    gert::TypedContinuousVector<size_t>* workspaceSize_;
    int64_t* tilingCond_;
    uint8_t* scheduleMode_;
    uint32_t* dynUBufSize_;
    ExpandableRtsArgBuffer* rtsArgBuffer_;

    size_t inputNum_;
    size_t outputNum_;
};

constexpr size_t MAX_WORKSPACE_NUM = 64; // max number of op workspace
// launch_arg 初始容量
constexpr size_t LAUNCH_ARG_INIT_SIZE = 128 * 1024;
// tiling_host_data 初始容量
constexpr size_t TILING_HOST_DATA_INIT_SIZE = 800 * 1024;

// MAX_ATTR_STRING_SIZE limit has been removed. String attr capacity is now dynamically managed.
// The value is kept for reference only.
constexpr size_t MAX_ATTR_STRING_SIZE = 1024; // Reference: max op attr string length (limit removed)
// ATTR_CAPACITY is used as the initial capacity for attr storage.
// When attr data exceeds this size, it will automatically expand (2x growth factor).
constexpr size_t ATTR_CAPACITY = 32 * 1024; // Initial attr capacity: max total op attr data size in Bytes
// MAX_OP_ARG_NUM is used as the initial capacity for op args (inputs/outputs).
// When arg count exceeds this value, it will automatically expand (2x growth factor).
constexpr size_t MAX_OP_ARG_NUM = 1024;    // Initial arg capacity: max op input/output arg count
constexpr size_t MAX_OP_TYPE_COUNT = 1024; // max registed op type
} // namespace op::internal

#endif
