/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAUNCH_ARG_INFO_H_
#define LAUNCH_ARG_INFO_H_

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "aclnn/acl_meta.h"
#include "kernel_utils.h"
#include "nlohmann/json.hpp"
#include "opdev/op_arg_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"

namespace op {
namespace internal {
enum class FormatType { NOT_SUPPORT_ND, SUPPORT_ND, ONLY_SUPPORT_ND, BOTTOM };

struct FormatInfo {
    std::unordered_set<ge::Format> supportFormats;
    FormatType fmtType = FormatType::BOTTOM;
};

enum class DtMatchMode {
    NORMAL,     // data type need to be exactly the same with binary json
    DTYPE_BYTE, // can be matched if data type width is the same. for example uint8 & int8; int16 & float16
    BOTTOM
};

enum class TensorType {
    REQUIRED,
    OPTIONAL,
    DYNAMIC,
    DYNAMIC_FOLDED, // pointer to pointer to tensor list
    BOTTOM,
};

enum class ShapeSupportType {
    SUPPORT_ALL,
    NOT_SUPPORT_ALL,
    DEFAULT,
};

constexpr size_t INVALID_MAX_DIM_NUM = gert::Shape::kMaxDimNum + 1;
struct TensorInfo {
    FormatInfo fmtInfo;
    DtMatchMode dtMatchMode = DtMatchMode::NORMAL;
    TensorType tensorType = TensorType::BOTTOM;
    uint32_t opType = INVALID_OP_TYPE_ID;
    ShapeSupportType shapeSupportType = ShapeSupportType::DEFAULT;
    size_t shapeDimension = INVALID_MAX_DIM_NUM;
};

constexpr size_t MAX_ATTR_SIZE = 32;
struct AttrInfo {
    std::string attrName;
    bool supportAll = false;
    /* Attributes can only be int64, float, bool, string, list_float, list_int64, list_list_int64 */
    size_t realSize = 0;
    std::unique_ptr<uint8_t[]> values = nullptr;
};

constexpr size_t MAX_TENSOR_SIZE = 32;
constexpr size_t MAX_VALID_DTYPE_SIZE = 16;

constexpr size_t FRACTAL_Z_SIZE = 4;

constexpr wchar_t MAX_VALID_DTYPE_FORMAT_KEY = static_cast<wchar_t>(ge::FORMAT_END);
constexpr wchar_t LEFT_BRACKET = '[';
constexpr wchar_t RIGHT_BRACKET = ']';
constexpr wchar_t SLASH = '/';
constexpr wchar_t COMMA = ',';

constexpr size_t PTR_OFFSET_SIZE = 8; // ptr_offset(8B)
constexpr size_t PRT_DIM_SIZE = 8;    // dim(4B) + cnt(4B)
constexpr size_t ARG_VECTOR_CAP_SIZE = 32;

class LaunchArgInfo {
public:
    LaunchArgInfo(bool genPlaceholder, bool hasDevPtrArg, OpArgContext* args)
        : allArg_(addrInfo), genPlaceholder_(genPlaceholder), hasDevPtrArg_(hasDevPtrArg)
    {
        allArg_.reserve(ARG_VECTOR_CAP_SIZE);
        if (args->ContainsOpArgType(OpArgDef::OP_INPUT_ARG)) {
            args->GetOpArg(OpArgDef::OP_INPUT_ARG)->VisitByNoReturn([this]([[maybe_unused]] size_t idx, OpArg& elem) {
                this->AppendInputLaunchArg(elem);
            });
        }
        if (args->ContainsOpArgType(OpArgDef::OP_OUTPUT_ARG)) {
            args->GetOpArg(OpArgDef::OP_OUTPUT_ARG)->VisitByNoReturn([this]([[maybe_unused]] size_t idx, OpArg& elem) {
                this->AppendLaunchArg(elem);
            });
        }
        if (args->ContainsOpArgType(OpArgDef::OP_OUTSHAPE_ARG)) {
            this->AppendOutshapeLaunchArg(
                reinterpret_cast<aclTensor*>((*args->GetOpArg(OpArgDef::OP_OUTSHAPE_ARG))[0]->pointer));
        }
        if (args->ContainsOpArgType(OpArgDef::OP_WORKSPACE_ARG)) {
            args->GetOpArg(OpArgDef::OP_WORKSPACE_ARG)
                ->VisitByNoReturn(
                    [this]([[maybe_unused]] size_t idx, OpArg& elem) { this->AppendWorkspaceLaunchArg(elem); });
        }
    }

    size_t GetDevArgNum() const { return devArgNum_; }

    size_t GetHostArgNum() const { return hostArgNum_; }

    size_t GetTensorNum() const { return tensorNum_; }

    uint32_t GetDFXInfoDumpElemCount() const { return dfxInfoDumpElemCount_; }

    void* GetDFXInfoDumpAddr() const { return dfxInfoDumpAddr_; }

    void SetDFXInfoDumpAddr(void* dumpAddr) { dfxInfoDumpAddr_ = dumpAddr; }

    size_t GetDFXInfoOffsetInTilingData() const { return dfxInfoOffsetInTilingData_; }

    void SetDFXInfoOffsetInTilingData(size_t offset) { dfxInfoOffsetInTilingData_ = offset; }

    struct DevArgInfo {
        const aclTensor* tensor;
        void* devAddr;
    };
    struct HostArgInfo {
        void* hostAddr;
        size_t hostDataLen;
    };
    struct DevPtrArgInfo {
        const aclTensorList* tensors;
        size_t ptrListLen;
    };
    class ArgAddr {
    public:
        explicit ArgAddr(const aclTensor* tensor, void* devAddr, bool isOutShapeTensor = false)
            : tag_(ArgTag::DEVICE_ARG),
              devAddr_({tensor, devAddr}),
              hostTensor_(nullptr),
              isOutShapeTensor_(isOutShapeTensor) {};
        explicit ArgAddr(void* hostAddr, size_t hostDataLen, const aclTensor* tensor = nullptr)
            : tag_(ArgTag::HOST_ARG),
              hostAddr_({hostAddr, hostDataLen}),
              hostTensor_(tensor),
              isOutShapeTensor_(false) {};
        explicit ArgAddr(const aclTensorList* tensors, size_t ptrListLen)
            : tag_(ArgTag::DEVICE_PTR_ARG),
              devPtrAddr_({tensors, ptrListLen}),
              hostTensor_(nullptr),
              isOutShapeTensor_(false) {};
        enum class ArgTag { DEVICE_ARG, HOST_ARG, DEVICE_PTR_ARG };
        ArgTag tag_;
        union {
            DevArgInfo devAddr_;
            HostArgInfo hostAddr_;
            DevPtrArgInfo devPtrAddr_;
        };
        const aclTensor* hostTensor_;
        bool isOutShapeTensor_;
    };

    const std::vector<ArgAddr>& GetAllArgInfo() const { return allArg_; }

    size_t GetArgSize() const { return argSize_; }

    size_t GetFirstWorkspaceIdx() const { return firstWorkspaceIdx_; }

private:
    std::vector<ArgAddr>& allArg_;
    size_t argSize_{0};
    size_t devArgNum_{0};
    size_t hostArgNum_{0};
    size_t tensorNum_{0};
    uint32_t dfxInfoDumpElemCount_{0};
    void* dfxInfoDumpAddr_{nullptr};
    size_t dfxInfoOffsetInTilingData_{0};
    bool genPlaceholder_{false};
    bool hasDevPtrArg_{false};
    size_t firstWorkspaceIdx_{0}; // workspace cant be first param for any ops

    static constexpr uint32_t DFX_INFO_SCALAR_ELEM_COUNT = 3U; // 标量tensor固定占用3个uint64_t
    static constexpr uint32_t DFX_INFO_FIXED_ELEM_COUNT = 2U;  // 每个tensor固定占用2个uint64_t(size + dimNum)

    inline uint32_t CalAdumpDFXInfoElemCount(const aclTensor* tensor) const
    {
        if (tensor == nullptr) {
            return 0;
        }
        uint32_t elemCount = 0;
        size_t dimNum = tensor->GetStorageShape().GetDimNum();
        if (dimNum == 0) {
            elemCount = DFX_INFO_SCALAR_ELEM_COUNT;
        } else {
            elemCount = DFX_INFO_FIXED_ELEM_COUNT + static_cast<uint32_t>(dimNum);
        }
        return elemCount;
    }

    void AddDeviceArg(const aclTensor* tensor, void* devAddr, bool isOutShapeTensor = false)
    {
        if (argSize_ >= allArg_.size()) {
            allArg_.emplace_back(tensor, devAddr, isOutShapeTensor);
        } else {
            ArgAddr& argAddr = allArg_.at(argSize_);
            argAddr.devAddr_.tensor = tensor;
            argAddr.devAddr_.devAddr = devAddr;
            argAddr.tag_ = ArgAddr::ArgTag::DEVICE_ARG;
            argAddr.hostTensor_ = nullptr;
            argAddr.isOutShapeTensor_ = isOutShapeTensor;
        }
        argSize_++;
    }

    void AddHostArg(void* hostAddr, size_t hostDataLen, const aclTensor* tensor = nullptr)
    {
        if (argSize_ >= allArg_.size()) {
            allArg_.emplace_back(hostAddr, hostDataLen, tensor);
        } else {
            ArgAddr& argAddr = allArg_.at(argSize_);
            argAddr.hostAddr_.hostAddr = hostAddr;
            argAddr.hostAddr_.hostDataLen = hostDataLen;
            argAddr.tag_ = ArgAddr::ArgTag::HOST_ARG;
            argAddr.hostTensor_ = tensor;
            argAddr.isOutShapeTensor_ = false;
        }
        argSize_++;
    }

    void AddDevicePtrArg(const aclTensorList* tensors, size_t ptrListLen)
    {
        if (argSize_ >= allArg_.size()) {
            allArg_.emplace_back(tensors, ptrListLen);
        } else {
            ArgAddr& argAddr = allArg_.at(argSize_);
            argAddr.devPtrAddr_.tensors = tensors;
            argAddr.devPtrAddr_.ptrListLen = ptrListLen;
            argAddr.tag_ = ArgAddr::ArgTag::DEVICE_PTR_ARG;
            argAddr.hostTensor_ = nullptr;
            argAddr.isOutShapeTensor_ = false;
        }
        argSize_++;
    }

    void AppendInputLaunchArg(const aclTensor* arg)
    {
        if (arg == nullptr) {
            // This is somewhat inconsitence here: null inputs are needed in op tiling, but no in kernel larunching.
            OP_LOGW("Append Launch NULL aclTensor");
            if (genPlaceholder_) {
                AddDeviceArg(nullptr, nullptr);
                devArgNum_++;
                tensorNum_++;
                dfxInfoDumpElemCount_++;
                OP_LOGW("Append Launch NULL aclTensor placeholder");
            }
            return;
        }
        if (arg->GetPlacement() == gert::kOnDeviceHbm) {
            AddDeviceArg(arg, arg->GetData());
            devArgNum_++;
            tensorNum_++;
#ifdef DEBUG
            OP_LOGD("Append Launch DEVICE aclTensor. dims: %zu, dtype: %d, aclTensor: %p, addr: %p, tensor size: %zu",
                    arg->GetStorageShape().GetDimNum(), arg->GetDataType(), arg, arg->GetData(),
                    arg->GetTensor()->GetSize());
            const auto& shape = arg->GetStorageShape();
            for (size_t i = 0; i < shape.GetDimNum(); i++) {
                OP_LOGD("#### DEVICE aclTensor Shape: %lu", shape.GetDim(i));
            }
#endif
        } else {
            size_t size = arg->Size() * ge::GetSizeByDataType(arg->GetDataType());
            AddHostArg(arg->GetData(), size, arg);
            hostArgNum_++;
            tensorNum_++;
#ifdef DEBUG
            OP_LOGD("Append Launch HOST aclTensor. dims: %zu, dtype: %d, %p, calc size: %zu, tensor size: %zu",
                    arg->GetStorageShape().GetDimNum(), arg->GetDataType(), arg, size, arg->GetTensor()->GetSize());
            const auto& shape = arg->GetStorageShape();
            for (size_t i = 0; i < shape.GetDimNum(); i++) {
                OP_LOGD("#### HOST aclTensor Shape: %lu", shape.GetDim(i));
            }
#endif
        }
        dfxInfoDumpElemCount_ += CalAdumpDFXInfoElemCount(arg);
    }

    void AppendInputLaunchArg(const aclTensorList* arg)
    {
        if (arg == nullptr) {
            OP_LOGW("aclTensorList is nullptr.");
            return;
        }
        if (hasDevPtrArg_) {
            size_t dataSize = PTR_OFFSET_SIZE;
            for (uint64_t i = 0; i < arg->Size(); i++) {
                dataSize += PRT_DIM_SIZE;
                if ((*arg)[i] != nullptr) {
                    dataSize += (*arg)[i]->GetStorageShape().GetDimNum() * sizeof(int64_t);
                }
            }
            AddDevicePtrArg(arg, dataSize);
            devArgNum_++;
            tensorNum_ += arg->Size() + 1;
            dfxInfoDumpElemCount_++;
#ifdef DEBUG
            OP_LOGD("Append input DEVICE ptr: size %lu, dataSize %zu", arg->Size(), dataSize);
#endif
            return;
        }
        for (uint64_t i = 0; i < arg->Size(); i++) {
            AppendInputLaunchArg((*arg)[i]);
        }
    }

    void AppendInputLaunchArg(OpArg& arg)
    {
        if (arg.type == OpArgType::OPARG_ACLTENSOR) {
            AppendInputLaunchArg(reinterpret_cast<aclTensor*>(arg->pointer));
        } else if (arg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
            AppendInputLaunchArg(reinterpret_cast<aclTensorList*>(arg->pointer));
        }
    }

    void AppendLaunchArg(const aclTensor* arg)
    {
        if (arg == nullptr) {
            OP_LOGW("Append Launch NULL aclTensor");
            return;
        }
        AddDeviceArg(arg, arg->GetData());
        devArgNum_++;
        tensorNum_++;
        dfxInfoDumpElemCount_ += CalAdumpDFXInfoElemCount(arg);
#ifdef DEBUG
        OP_LOGD("Append Launch aclTensor. dims: %zu, dtype: %d, data:%p, tensor size: %zu",
                arg->GetStorageShape().GetDimNum(), arg->GetDataType(), arg->GetData(), arg->GetTensor()->GetSize());
        const auto& shape = arg->GetStorageShape();
        for (size_t i = 0; i < shape.GetDimNum(); i++) {
            OP_LOGD("#### DEVICE aclTensor Shape: %lu", shape.GetDim(i));
        }
#endif
    }

    void AppendLaunchArg(const aclTensorList* arg)
    {
        if (arg == nullptr) {
            return;
        }
        if (hasDevPtrArg_) {
            size_t dataSize = PTR_OFFSET_SIZE;
            for (uint64_t i = 0; i < arg->Size(); i++) {
                dataSize += PRT_DIM_SIZE;
                if ((*arg)[i] != nullptr) {
                    dataSize += (*arg)[i]->GetStorageShape().GetDimNum() * sizeof(int64_t);
                }
            }
            AddDevicePtrArg(arg, dataSize);
            devArgNum_++;
            tensorNum_ += arg->Size() + 1;
            dfxInfoDumpElemCount_++;
#ifdef DEBUG
            OP_LOGD("Append output DEVICE ptr: size %ld, dataSize %ld", arg->Size(), dataSize);
#endif
            return;
        }
        for (uint64_t i = 0; i < arg->Size(); i++) {
            AppendLaunchArg((*arg)[i]);
        }
    }

    void AppendLaunchArg(OpArg& arg)
    {
        if (arg.type == OpArgType::OPARG_ACLTENSOR) {
            AppendLaunchArg(reinterpret_cast<aclTensor*>(arg->pointer));
        } else if (arg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
            AppendLaunchArg(reinterpret_cast<aclTensorList*>(arg->pointer));
        }
    }

    void AppendOutshapeLaunchArg(const aclTensor* arg)
    {
        AddDeviceArg(arg, arg->GetData(), true);
        devArgNum_++;
        tensorNum_++;
        dfxInfoDumpElemCount_++;
#ifdef DEBUG
        OP_LOGD("Append Launch Outshape Tensor. dims: %zu, dtype: %d, data:%p", arg->GetStorageShape().GetDimNum(),
                arg->GetDataType(), arg->GetData());
        const auto& shape = arg->GetStorageShape();
        for (size_t i = 0; i < shape.GetDimNum(); i++) {
            OP_LOGD("#### DEVICE aclTensor Shape: %lu", shape.GetDim(i));
        }
#endif
    }

    void AppendWorkspaceLaunchArg(const aclTensor* arg) { AppendLaunchArg(arg); }

    void AppendWorkspaceLaunchArg(const aclTensorList* arg)
    {
        if (arg != nullptr) {
            for (uint64_t i = 0; i < arg->Size(); i++) {
                AppendLaunchArg((*arg)[i]);
            }
        }
    }

    void AppendWorkspaceLaunchArg(const std::vector<std::tuple<void*, const aclTensor*>>& arg)
    {
        for (const auto& elem : arg) {
#ifdef DEBUG
            OP_LOGD("Append Launch Memset Tensor. tensor: %p, data:%p", std::get<1>(elem), std::get<0>(elem));
#endif
            if (std::get<1>(elem) == nullptr) {
                OP_LOGW("Memset tensor is nullptr.");
            }
            AddDeviceArg(std::get<1>(elem), std::get<0>(elem));
            devArgNum_++;
            tensorNum_++;
            dfxInfoDumpElemCount_ += CalAdumpDFXInfoElemCount(std::get<1>(elem));
        }
    }

    void AppendWorkspaceLaunchArg(OpArg& arg)
    {
        if (firstWorkspaceIdx_ == 0) {
            firstWorkspaceIdx_ = argSize_;
        }
        if (arg.type == OpArgType::OPARG_ACLTENSOR) {
            AppendLaunchArg(reinterpret_cast<aclTensor*>(arg->pointer));
        } else if (arg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
            AppendWorkspaceLaunchArg(reinterpret_cast<aclTensorList*>(arg->pointer));
        } else if (arg.type == OpArgType::OPARG_MEMSET_WORKSPACE) {
            AppendWorkspaceLaunchArg(
                *reinterpret_cast<std::vector<std::tuple<void*, const aclTensor*>>*>(arg->pointer));
        }
    }

    thread_local static std::vector<ArgAddr> addrInfo;
};
} // namespace internal
} // namespace op

#endif
