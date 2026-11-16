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
 * \file loopbuf.h
 * \brief
 */
#ifndef UTIL_LOOPBUF_H_
#define UTIL_LOOPBUF_H_

#include "atvoss/util/kernel_vec_intf_apt.h"
namespace Ops {
namespace Base {
#ifdef __CCE_KT_TEST__
#define PRINT_ARRAY(arr, size, str)                                      \
    do {                                                                 \
        for (int i = 0; i < size; i++) {                                 \
            auto wrapper = arr[i];                                       \
            RUN_LOG("Init %s[%d], bufID %d \n", str, i, wrapper->bufID); \
        }                                                                \
    } while (0)
#else
#define PRINT_ARRAY(arr, size, str)
#endif

const static int DIGITAL_TWO = 2;
struct TBufferWrapper {
    TEventID bufID;                    // buf的ID号
    TBufState state = TBufState::FREE; // 当前状态
    TBuf<TPosition::VECCALC> buf;      // tbuf空间

public:
    __aicore__ inline TBufferWrapper() = default;
};

// buf_       |ping mte2|ping mte3|pong mte2|pong mte3|tmpBuf|
// pingPtr_   |ping mte2|tmpBuf|pong mte3|
// pongPtr_   |pong mte2|tmpBuf|ping mte3|
template <int mte2Count, int mte3Count, int bufCount, int mode>
struct TLoopBuffer {
public:
    __aicore__ inline TLoopBuffer() = default;

protected:
    // 创建buf数组以及存放buf指针的ping pong指针数组
    struct TBufferWrapper buf_[bufCount];
    struct TBufferWrapper* pingPtr_[bufCount - mte2Count - mte3Count];
    struct TBufferWrapper* pongPtr_[bufCount - mte2Count - mte3Count];

    // 计算tmpbuf的大小，mte2以及mte3的位置
    constexpr static int tmpCount = bufCount - DIGITAL_TWO * mte2Count - DIGITAL_TWO * mte3Count;
    constexpr static int mte3Posion = mte2Count + tmpCount;
    constexpr static int mte2Posion = mte2Count;
    constexpr static int mteCount = mte2Count + mte3Count;
    constexpr static int maxCount = mte2Count + mte3Count + tmpCount;

    // pingPtr_和pongPtr_数组记录读写位置的指针
    int pingInCursor = 0;
    int pongInCursor = 0;
    int pingCursor = mte2Count;
    int pongCursor = mte2Count;
    int pingOutCursor = mte3Posion;
    int pongOutCursor = mte3Posion;

public:
    /**
     * 调用get_buf函数
     * @tparam p TPosition的类型
     * @param bufID 当前处理的buf的真是ID
     */
    template <TPosition p>
    __aicore__ inline void GetBuf(TEventID bufID)
    {
        if constexpr (p == TPosition::VECIN) {
            Mutex::Lock<PIPE_MTE2>(bufID);
        } else if constexpr (p == TPosition::VECCALC) {
            Mutex::Lock<PIPE_V>(bufID);
        } else if constexpr (p == TPosition::VECOUT) {
            Mutex::Lock<PIPE_MTE3>(bufID);
        }
    }

    /**
     * 调用rls_buf函数
     * @tparam p TPosition的类型
     * @param bufID 当前处理的buf的真实ID
     */
    template <TPosition p>
    __aicore__ inline void RlsBuf(TEventID bufID)
    {
        if constexpr (p == TPosition::VECIN) {
            Mutex::Unlock<PIPE_MTE2>(bufID);
        } else if constexpr (p == TPosition::VECCALC) {
            Mutex::Unlock<PIPE_V>(bufID);
        } else if constexpr (p == TPosition::VECOUT) {
            Mutex::Unlock<PIPE_MTE3>(bufID);
        }
    }

    /**
     * 进行初始化
     * @param blockLen 每一块计算申请buf的大小
     * @param pipe 全局TPipe指针
     */
    __aicore__ inline void Init(int blockLen, TPipe* pipe)
    {
        RUN_LOG("mte2Count: %d, mte3Count: %d, bufCount: %d, mode: %d", mte2Count, mte3Count, bufCount, mode);
        for (int i = 0; i < bufCount; i++) {
            auto wrapper = &(buf_[i]);
            wrapper->bufID = i;
            pipe->InitBuffer(wrapper->buf, blockLen);
            if (i < mte2Count) {
                pingPtr_[i] = wrapper;
            } else if (i < mteCount) {
                pongPtr_[i + tmpCount] = wrapper;
            } else if (i < mteCount + mte2Count) {
                pongPtr_[i - mteCount] = wrapper;
            } else if (i < mteCount * DIGITAL_TWO) {
                pingPtr_[i - mteCount + tmpCount] = wrapper;
            } else {
                pingPtr_[i - mteCount - mte3Count] = wrapper;
                pongPtr_[i - mteCount - mte3Count] = wrapper;
            }
        }
        PRINT_ARRAY(this->pingPtr_, maxCount, "PingPtr_");
        PRINT_ARRAY(this->pongPtr_, maxCount, "PongPtr_");
    }

    /**
     * 选择空闲的buffer空间
     * @tparam count 在队列寻找的次数
     * @tparam startPosion 本次循环结束，下一次初始的位置
     * @tparam endPosion 当前循环结束的位置
     * @tparam initPosion 寻找到可用空间后，重置初始化的位置
     * @param ptr 当前寻找的buf指针数组
     * @param cursor 当前操作记录使用的指针
     * @return TBufferWrapper对象
     */
    template <int count, int startPosion, int endPosion, int initPosion>
    __aicore__ inline TBufferWrapper* GetWrapper(TBufferWrapper** ptr, int& cursor)
    {
        TBufferWrapper* wrapper;
        for (int i = 0; i < count; i++) {
            wrapper = ptr[cursor];
            cursor += 1;
            cursor = (cursor >= endPosion) ? startPosion : cursor;
            if (wrapper->state == TBufState::FREE) {
                cursor = initPosion;
                return wrapper;
            }
        }
    }

    /**
     * 数据从mte2搬运到ub操作，选择合适buffer的函数
     * @tparam status 0代表ping 1代表pong
     * @return TBufferWrapper对象
     */
    template <int status>
    __aicore__ inline TBufferWrapper* GetInWrapper()
    {
        TBufferWrapper* wrapper;
        if constexpr (status == 0) { // ping操作
            if constexpr (mode == 2) {
                wrapper = this->pingPtr_[this->pingInCursor];
                this->pingInCursor++;
                this->pingInCursor = (this->pingInCursor >= mte2Posion) ? 0 : this->pingInCursor;
            } else if constexpr (mode == 1) {
                wrapper = GetWrapper<maxCount, 0, mte3Posion, 0>(this->pingPtr_, this->pingInCursor);
            } else if constexpr (mode == 0) {
                wrapper = GetWrapper<maxCount, 0, maxCount, 0>(this->pingPtr_, this->pingInCursor);
            }
        } else {
            if constexpr (mode == 2) {
                wrapper = this->pongPtr_[this->pongInCursor];
                this->pongInCursor++;
                this->pongInCursor = (this->pongInCursor >= mte2Posion) ? 0 : this->pongInCursor;
            } else if constexpr (mode == 1) {
                wrapper = GetWrapper<maxCount, 0, mte3Posion, 0>(this->pongPtr_, this->pongInCursor);
            } else if constexpr (mode == 0) {
                wrapper = GetWrapper<maxCount, 0, maxCount, 0>(this->pongPtr_, this->pongInCursor);
            }
        }
        return wrapper;
    }

    /**
     * 数据从UB搬运到MTE3操作，选择合适buffer的函数
     * @tparam status 0代表ping 1代表pong
     * @return TBufferWrapper对象
     */
    template <int status>
    __aicore__ inline TBufferWrapper* GetOutWrapper()
    {
        TBufferWrapper* wrapper;
        if constexpr (status == 0) {
            if constexpr (mode == 1 || mode == 2) {
                wrapper = this->pongPtr_[this->pingOutCursor];
                this->pingOutCursor++;
                this->pingOutCursor = (this->pingOutCursor >= maxCount) ? mte3Posion : this->pingOutCursor;
            } else {
                for (int iter = mte3Posion; iter < maxCount; iter++) {
                    wrapper = pongPtr_[iter];
                    if (wrapper->state == TBufState::FREE) {
                        return wrapper;
                    }
                }
                wrapper = GetWrapper<maxCount, 0, maxCount, mte2Posion>(this->pingPtr_, this->pingCursor);
            }
        } else {
            if constexpr (mode == 1 || mode == 2) {
                wrapper = this->pingPtr_[this->pongOutCursor];
                this->pongOutCursor++;
                this->pongOutCursor = (this->pongOutCursor >= maxCount) ? mte3Posion : this->pongOutCursor;
            } else {
                for (int iter = mte3Posion; iter < maxCount; iter++) {
                    wrapper = pingPtr_[iter];
                    if (wrapper->state == TBufState::FREE) {
                        return wrapper;
                    }
                }
                wrapper = GetWrapper<maxCount, 0, maxCount, mte2Posion>(this->pongPtr_, this->pongCursor);
            }
        }
        return wrapper;
    }

    /**
     * 数据在UB内进行Vector操作，选择合适buffer的函数
     * @tparam status 0代表ping 1代表pong
     * @return TBufferWrapper对象
     */
    template <int status>
    __aicore__ inline TBufferWrapper* GetVecWrapper()
    {
        TBufferWrapper* wrapper;
        if constexpr (status == 0) {
            if constexpr (mode == 0) {
                wrapper = GetWrapper<maxCount + 1, 0, maxCount, mte2Posion>(this->pingPtr_, this->pingCursor);
            } else if constexpr (mode == 1) {
                wrapper = GetWrapper<mte3Posion + 1, 0, mte3Posion, mte2Posion>(this->pingPtr_, this->pingCursor);
            } else if constexpr (mode == 2) {
                wrapper = GetWrapper<tmpCount + 1, mte2Posion, mte3Posion, mte2Posion>(this->pingPtr_,
                                                                                       this->pingCursor);
            }
        } else {
            if constexpr (mode == 0) {
                wrapper = GetWrapper<maxCount + 1, 0, maxCount, mte2Posion>(this->pongPtr_, this->pongCursor);
            } else if constexpr (mode == 1) {
                wrapper = GetWrapper<mte3Posion + 1, 0, mte3Posion, mte2Posion>(this->pongPtr_, this->pongCursor);
            } else if constexpr (mode == 2) {
                wrapper = GetWrapper<tmpCount + 1, mte2Posion, mte3Posion, mte2Posion>(this->pongPtr_,
                                                                                       this->pongCursor);
            }
        }
        return wrapper;
    }

    /**
     * 申请合适buffer的主函数
     * @tparam p  TPosition的类型
     * @tparam isOut  true代表连接搬出操作的内存申请 false代表输入或者VEctor操作的内存申请
     * @param state 0代表ping 1代表pong
     * @return TBufferWrapper对象
     */
    template <TPosition p, bool isOut>
    __aicore__ inline TBufferWrapper* AllocTensor(int state)
    {
        TBufferWrapper* wrapper;
        if (state == 0) {                          // Ping
            if constexpr (p == TPosition::VECIN) { // 处理输入
                wrapper = GetInWrapper<0>();
            } else if constexpr (isOut) { // 处理输出
                wrapper = GetOutWrapper<0>();
            } else { // vec计算
                wrapper = GetVecWrapper<0>();
            }
        } else { // Pang
            if constexpr (p == TPosition::VECIN) {
                wrapper = GetInWrapper<1>();
            } else if constexpr (isOut) {
                wrapper = GetOutWrapper<1>();
            } else {
                wrapper = GetVecWrapper<1>();
            }
        }
        wrapper->state = TBufState::OCCUPIED;
        if constexpr (p == TPosition::VECIN || isOut) {
            GetBuf<p>(wrapper->bufID);
        }
        RUN_LOG("AllocBuf: ID: %d", wrapper->bufID);
        return wrapper;
    }

    /**
     * 给VECIN或者VECOUT空间插入get_buf同步的函数
     * @tparam p  TPosition的类型
     * @param wrapper TBufferWrapper对象
     */
    template <TPosition p>
    __aicore__ void inline GetTensor(TBufferWrapper* wrapper)
    {
        RUN_LOG("GetTensor: ID: %d", wrapper->bufID);
        GetBuf<p>(wrapper->bufID);
    }

    /**
     * 给VECIN或者VECOUT空间插入rls_buf同步的函数
     * @tparam p  TPosition的类型
     * @param wrapper TBufferWrapper对象
     */
    template <TPosition p>
    __aicore__ void inline ReleaseTensor(TBufferWrapper* wrapper)
    {
        RUN_LOG("ReleaseTensor: ID: %d", wrapper->bufID);
        RlsBuf<p>(wrapper->bufID);
    }

    /**
     * 对空间进行状态上的释放，以便后续复用
     * @param wrapper TBufferWrapper对象
     */
    __aicore__ inline void FreeTensor(TBufferWrapper* wrapper)
    {
        RUN_LOG("FreeTensor: ID: %d", wrapper->bufID);
        wrapper->state = TBufState::FREE;
    }
};

} // namespace Base
} // namespace Ops

#endif // UTIL_LOOPBUF_H_