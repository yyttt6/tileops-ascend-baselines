/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_tensor_empty.h
 * \brief reduce_tensor_empty kernel
 */

#ifndef REDUCE_TENSOR_EMPTY_H
#define REDUCE_TENSOR_EMPTY_H

namespace Ops {
namespace Base {
namespace ReduceOpTmpl {
using namespace Ops::Base;

template <typename T>
class ReduceTensorEmpty {
public:
    template <class... Args>
    __aicore__ inline explicit ReduceTensorEmpty(const ReduceOpTilingData* tilingData, GlobalTensor<uint8_t>* output,
                                                 TPipe* pipeIn, Args... args)
    {
        tiling_ = tilingData;
        bufferSize_ = tiling_->basicBlock == 0 ? bufferSize_ : tiling_->basicBlock;
        output_ = output;

        pipeIn->InitBuffer(dupTempBuf_, UB_SIZE);
        pipeIn->InitBuffer(vecQue_, BUFFER_NUM, bufferSize_);
        InitValue(args...);
    };

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

    template <class T1, class... Args>
    __aicore__ inline void InitValue(T1 value, Args... args)
    {
        constexpr uint32_t UB_SIZE = Ops::Base::GetUbBlockSize();
        if constexpr (IsSameType<T, T1>::value) {
            value_ = value;
        } else {
            auto tempBuf = dupTempBuf_.Get<T1>();
            Duplicate<T1>(tempBuf, value, UB_SIZE / sizeof(T1));
            auto tempCastBuf = tempBuf.template ReinterpretCast<T>();
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                Cast(tempCastBuf, tempBuf, RoundMode::CAST_RINT, 1);
            } else {
                Cast(tempCastBuf, tempBuf, RoundMode::CAST_NONE, 1);
            }
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            value_ = tempCastBuf.GetValue(0);
        }
    }

    __aicore__ inline void InitValue() {};

private:
    constexpr static uint32_t UB_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t BUFFER_NUM = 2;
    const ReduceOpTilingData* tiling_;

    GlobalTensor<uint8_t>* output_;
    TBuf<TPosition::VECCALC> dupTempBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> vecQue_;

    int64_t bufferSize_ = 64 * 1024;
    uint64_t loopStartIdx_ = 0;
    uint64_t loopEndIdx_ = 0;
    T value_ = 0;

    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void ReduceTensorEmpty<T>::Process()
{
    if (tiling_->outSize <= 0) {
        return;
    }
    int64_t blockIdx = GetBlockIdx();
    loopStartIdx_ = blockIdx * tiling_->factorACntPerCore;
    loopEndIdx_ = loopStartIdx_ + tiling_->factorACntPerCore;
    if (unlikely(loopEndIdx_ > tiling_->factorATotalCnt)) {
        loopEndIdx_ = tiling_->factorATotalCnt;
    }

    ProcessPerCore();
}

template <typename T>
__aicore__ inline void ReduceTensorEmpty<T>::ProcessPerCore()
{
    int64_t copyElementNum = tiling_->ubFactorA;
    for (uint64_t loopIdx = loopStartIdx_; loopIdx < loopEndIdx_; loopIdx++) {
        if (loopIdx == tiling_->factorATotalCnt - 1) {
            copyElementNum = tiling_->outSize - (tiling_->factorATotalCnt - 1) * tiling_->ubFactorA;
        }

        auto bindLocalIn = vecQue_.AllocTensor<T>();
        copyOutParams_.blockLen = copyElementNum * sizeof(T);
        Duplicate<T>(bindLocalIn, value_, copyElementNum);
        vecQue_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(bindLocalIn);

        auto bindLocalOut = vecQue_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T>();
        GlobalTensor<T> globalTensor;
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output_[0].GetPhyAddr(0)));
        DataCopyPad(globalTensor[loopIdx * tiling_->ubFactorA], bindLocalOut, copyOutParams_);

        vecQue_.FreeTensor(bindLocalOut);
    }
}

} // namespace ReduceOpTmpl
} // namespace Base
} // namespace Ops
#endif // REDUCE_TENSOR_EMPTY_H