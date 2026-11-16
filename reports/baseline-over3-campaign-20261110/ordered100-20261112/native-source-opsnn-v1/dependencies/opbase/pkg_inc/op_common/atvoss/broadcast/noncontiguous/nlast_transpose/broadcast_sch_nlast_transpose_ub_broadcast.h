/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file broadcast_sch_nlast_transpose_ub_broadcast.h
 * \brief
 */

#ifndef BROADCAST_SCH_NLAST_TRANSPOSE_UB_BROADCAST_H_
#define BROADCAST_SCH_NLAST_TRANSPOSE_UB_BROADCAST_H_

#include "broadcast_sch_nlast_transpose_ub_broadcast_base.h"
#include "op_kernel/platform_util.h"

#pragma "lib"

namespace Ops {
namespace Base {
template <class BrcDag, int64_t R>
class BroadcastNlastTransposeUbBrcSch : public BroadcastNlastTransposeBaseSch<BrcDag, false, R> {
public:
    __aicore__ inline explicit BroadcastNlastTransposeUbBrcSch(
        const BroadcastNlastTransposeTilingData<BrcDag>* baseTilingData)
        : BroadcastNlastTransposeBaseSch<BrcDag, false, R>(baseTilingData), tilingData(baseTilingData)
    {}
    /**
     * 初始化BroadcastNlastTransposeUbBrcSch对象
     * @tparam Args 输入Args类型
     * @param pipe 流水线指针
     * @param args 输入args参数列表，需要匹配DAG图中PlaceHolder的顺序[In0, In1..., Out0, Out1...]
     */
    template <class... Args>
    __aicore__ inline void Init(TPipe* pipe, Args... args)
    {
        static_assert(BrcDag::InputSize + BrcDag::OutputSize == sizeof...(Args),
                      "BroadcastNlastTransposeUbBrcSch.Init args num should match DAG holders.");
        InitInputArgs<0>(args...); // 调用入参分析,input, output
        this->template SetScalar<0>(0);
        TBuf<TPosition::VECCALC> buf;
        this->blockEleNum_ = tilingData->elemNum;
        this->blockLen_ = this->blockEleNum_ * BrcDag::MaxDtypeBytes;
        pipe->InitBuffer(buf, this->blockLen_ * bufferNum);
        this->tensorPool_ = buf.Get<uint8_t>();
        // 根据原始shape获取
        if constexpr (BrcDag::VecBrcSize > 0 || BrcDag::CopyBrcSize > 0) {
            this->template GetUbBroadcastShapeInfo(tilingData->outputDims, this->dstUbFormerShape_);
            this->runningRank_ = tilingData->shapeLen - tilingData->ubSplitAxis;
            if (tilingData->outputDims[tilingData->ubSplitAxis] != 1) {
                this->ubFormer_ = tilingData->ubFormer;
                this->ubTail_ = tilingData->ubTail;
            } else {
                this->ubFormer_ = 1;
                this->ubTail_ = 1;
            }
        }
        if constexpr (BrcDag::VecBrcSize > 0) {
            this->template GetVecBrcInfo<0>();
        }
        if constexpr (BrcDag::CopyBrcSize > 0) {
            GetCopyInBrcInfo<0>();
        }
    }

    /**
     * 执行BroadcastNlastTransposeUbBrcSch
     */
    __aicore__ inline void Process()
    {
        ubLoopNum_ = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingData->blockTail :
                                                                            tilingData->blockFormer;
        int64_t axesIndices[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingData->blockFormer * AscendC::GetBlockIdx(), tilingData->outputDims,
                                tilingData->ubSplitAxis, tilingData->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum_; ubLoopIdx++) {
            this->vBrcIndex_ = 0;
            cBrcIndex_ = 0;
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingData->outputDims, tilingData->ubSplitAxis,
                                           tilingData->ubOuter);
            }

            int64_t ubSplitSize = axesIndices[tilingData->ubSplitAxis] == tilingData->ubOuter - 1 ?
                                      tilingData->ubTail :
                                      tilingData->ubFormer;
            // ub整循环处理的元素个数
            Run<0, false>(ubSplitSize, axesIndices, ubLoopIdx, ubLoopIdx & 1);
        }
    }

protected:
    // 初始化输出
    template <int start, class... Args>
    __aicore__ inline void InitOutputArgs(GM_ADDR y, Args... args)
    {
        this->outGm_[start] = y;
        if constexpr (start + 1 < BrcDag::OutputSize) {
            InitOutputArgs<start + 1>(args...);
        }
    }

    // 初始化输入
    template <int start, class... Args>
    __aicore__ inline void InitInputArgs(GM_ADDR x, Args... args)
    {
        this->inGm_[start] = x;
        if constexpr (start + 1 < BrcDag::InputSize) {
            InitInputArgs<start + 1>(args...);
        } else {
            InitOutputArgs<0>(args...);
        }
    }

    // 获取CopyInBrc信息
    template <int index = 0>
    __aicore__ inline void GetCopyInBrcInfo()
    {
        using copyInBrcOp = typename BrcDag::CopyBrcNodes::template At<index>;
        using inputType = typename copyInBrcOp::template FunInArgType<0>;
        this->template GetUbBroadcastShapeInfo(tilingData->inputBrcDims[index], copyInBrcFormerShape[index]);
        if (tilingData->inputBrcDims[index][tilingData->ubSplitAxis] != 1) {
            cBrcFormer[index] = tilingData->ubFormer;
            cBrcTail[index] = tilingData->ubTail;
        } else {
            cBrcFormer[index] = 1;
            cBrcTail[index] = 1;
        }
        this->template GetStridesWithPadThroughDims(tilingData->inputBrcDims[index], inputBrcStridesWithPad[index]);

        if constexpr (index + 1 < BrcDag::CopyBrcSize) {
            GetCopyInBrcInfo<index + 1>();
        }
    }

    /**
     * CopyInAndVecBrc函数调用的Broadcast函数
     * @tparam T 数据类型
     * @param dst 目的tensor
     * @param src 源tensor
     * @param realIdx 当前处理的索引
     */
    template <typename T>
    __aicore__ inline void CvecBrc(LocalTensor<T>& dst, LocalTensor<T>& src, int64_t realIdx, int pos)
    {
        BroadcastTiling runningTiling;

        if (this->runningRank_ < BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM) {
            uint32_t padded[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {1, 1, 1, 1};
            for (uint32_t i = 0; i < this->runningRank_; ++i) {
                padded[4 - this->runningRank_ + i] = this->dstUbFormerShapeLastPad_[i];
            }
            for (uint32_t i = 0; i < BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM; ++i) {
                this->dstUbFormerShapeLastPad_[i] = padded[i];
            }

            uint32_t brc_padded[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {1, 1, 1, 1};
            for (uint32_t i = 0; i < this->runningRank_; ++i) {
                brc_padded[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM - this->runningRank_ +
                           i] = copyInBrcFormerShapeWithLastPad[cBrcIndex_][i];
            }
            for (uint32_t i = 0; i < BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM; ++i) {
                copyInBrcFormerShapeWithLastPad[cBrcIndex_][i] = brc_padded[i];
            }
        }

        GetBroadcastTilingInfo<T, R>(R, this->dstUbFormerShapeLastPad_, copyInBrcFormerShapeWithLastPad[cBrcIndex_],
                                     false, runningTiling);
        Broadcast<T, R>(dst, src, this->dstUbFormerShapeLastPad_, copyInBrcFormerShapeWithLastPad[cBrcIndex_],
                        &runningTiling);
    }

    /**
     * CopyInAndVecBrc函数
     * @tparam Op CopyInBrc的Bind节点
     * @tparam pos CopyInBrc在FunList内的索引
     * @param ubSplitSize UB切分大小
     * @param axesIndices 当前处理的索引
     * @param ubLoopIdx 当前处理的循环变量
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename Op, int pos>
    __aicore__ inline void CopyInAndVecBrc(int64_t ubSplitSize,
                                           const int64_t (&axesIndices)[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM],
                                           int64_t ubLoopIdx, int32_t pingPong)
    {
        static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");
        using input = typename Op::InHolders::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        int64_t realIdx = (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter;

        if ((tilingData->inputBrcStrides[cBrcIndex_][tilingData->ubSplitAxis] == 0) && ubLoopIdx > 0 && realIdx > 0) {
            cBrcIndex_++;
            return;
        }
        // Prepare input args
        auto intList = this->template GetBufId<pos, true>(pingPong);
        LocalTensor<inputType> inTensor = this->tensorPool_[intList[1] * this->blockLen_]
                                              .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inTensor.SetBufferLen(this->blockEleNum_);
#endif
        GlobalTensor<inputType> globalTensor;
        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->inputBrcStrides[cBrcIndex_],
                                                tilingData->ubSplitAxis, tilingData->ubFormer);

        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(this->inGm_[input::Pos] + gmOffset * sizeof(inputType)));
        // Set getBuf
        GetTensor<TPosition::VECIN>(intList[1]);
        // Run copyIn
        if (realIdx == tilingData->ubOuter - 1) {
            this->dstUbFormerShape_[0] = this->ubTail_;
            copyInBrcFormerShape[cBrcIndex_][0] = cBrcTail[cBrcIndex_];
        } else {
            this->dstUbFormerShape_[0] = this->ubFormer_;
            copyInBrcFormerShape[cBrcIndex_][0] = cBrcFormer[cBrcIndex_];
        }

        for (int i = 0; i < BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM; i++) {
            copyInBrcFormerShapeWithLastPad[cBrcIndex_][i] = copyInBrcFormerShape[cBrcIndex_][i];
            this->dstUbFormerShapeLastPad_[i] = this->dstUbFormerShape_[i];
        }
        if (this->runningRank_ > 1) {
            copyInBrcFormerShapeWithLastPad[cBrcIndex_][this->runningRank_ -
                                                        1] = inputBrcStridesWithPad[cBrcIndex_]
                                                                                   [tilingData->shapeLen - NUM_TWO];
            this->dstUbFormerShapeLastPad_[this->runningRank_ -
                                           1] = inputBrcStridesWithPad[cBrcIndex_][tilingData->shapeLen - NUM_TWO];
        }
        ubSplitSize = copyInBrcFormerShape[cBrcIndex_][0];
        int64_t block_count = 1;
        int64_t src_stride = 0;
        int64_t dst_stride = 0;
        int64_t block_len = ubSplitSize;
        uint32_t loop1Size = 1;
        uint64_t loop1SrcStride = 0;
        uint64_t loop1DstStride = 0;
        uint32_t loop2Size = 1;
        uint64_t loop2SrcStride = 0;
        uint64_t loop2DstStride = 0;

        if (this->runningRank_ > 1) {
            block_count = ubSplitSize;
            block_len = copyInBrcFormerShape[cBrcIndex_][this->runningRank_ - 1];
            src_stride = tilingData->inputBrcStrides[cBrcIndex_][tilingData->shapeLen - NUM_TWO] - block_len;

            int64_t inputTypeAlignSize = blockSize / sizeof(inputType);
            int64_t blockLenInputTypeAlignNum = (block_len + inputTypeAlignSize - 1) / inputTypeAlignSize *
                                                inputTypeAlignSize;
            dst_stride = ((inputBrcStridesWithPad[cBrcIndex_][tilingData->shapeLen - NUM_TWO] -
                           blockLenInputTypeAlignNum) *
                          sizeof(inputType)) /
                         blockSize;
        }

        if (this->runningRank_ > NUM_TWO) {
            block_count = copyInBrcFormerShape[cBrcIndex_][this->runningRank_ - NUM_TWO];
            loop1Size = ubSplitSize;
            loop1SrcStride = tilingData->inputBrcStrides[cBrcIndex_][tilingData->shapeLen - NUM_THREE] *
                             sizeof(inputType);
            loop1DstStride = inputBrcStridesWithPad[cBrcIndex_][tilingData->shapeLen - NUM_THREE] * sizeof(inputType);
        }

        if (this->runningRank_ > NUM_THREE) {
            loop1Size = copyInBrcFormerShape[cBrcIndex_][this->runningRank_ - NUM_THREE];
            loop2Size = ubSplitSize;
            loop2SrcStride = tilingData->inputBrcStrides[cBrcIndex_][tilingData->shapeLen - NUM_FOUR] *
                             sizeof(inputType);
            loop2DstStride = inputBrcStridesWithPad[cBrcIndex_][tilingData->shapeLen - NUM_FOUR] * sizeof(inputType);
        }

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(block_count), static_cast<uint32_t>(block_len * sizeof(inputType)),
            static_cast<uint32_t>(src_stride * sizeof(inputType)), static_cast<uint32_t>(dst_stride), 0};
        AscendC::DataCopyPadExtParams<inputType> padParams{false, 0, 0, 0};

        AscendC::LoopModeParams LoopParams{loop1Size,      loop2Size,      loop1SrcStride,
                                           loop1DstStride, loop2SrcStride, loop2DstStride};
        AscendC::SetLoopModePara(LoopParams, DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<inputType>(inTensor, globalTensor, copyParams, padParams);
        AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        // Set rlsBuf
        ReleaseTensor<TPosition::VECIN>(intList[1]);

        LocalTensor<inputType> outTensor = this->tensorPool_[intList[0] * this->blockLen_]
                                               .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        outTensor.SetBufferLen(this->blockEleNum_);
#endif
        GetTensor<TPosition::VECCALC>(intList[0]);
        GetTensor<TPosition::VECCALC>(intList[1]);
        CvecBrc<inputType>(outTensor, inTensor, realIdx, pos);
        ReleaseTensor<TPosition::VECCALC>(intList[1]);
        ReleaseTensor<TPosition::VECCALC>(intList[0]);
        cBrcIndex_++;
    }

    /* *
     * 搬入Op
     * @tparam Op CopyIn的Bind节点
     * @tparam pos CopyIn在FunList内的索引
     * @param axesIndices 当前处理的索引
     * @param ubLoopIdx 当前处理的循环变量
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return void
     */
    template <typename Op, int pos>
    __aicore__ inline void CopyInMoveAlign(int64_t handleLength,
                                           const int64_t (&axesIndices)[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM],
                                           int64_t ubLoopIdx, int32_t pingPong)
    {
        static_assert(Op::InHolders::Size == 1, "CopyInMoveAlign input inHolders num should be 1.");
        using input = typename Op::InHolders::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        if constexpr (Op::IsScalarOp) {
            inputType scalar = this->template GetScalar<inputType, input>();
            this->opScalars_.template Set<pos>(scalar);
            return;
        }
        int64_t realIdx = (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter;

        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->inputStrides[input::Pos],
                                                tilingData->ubSplitAxis, tilingData->ubFormer);
        // Prepare input args
        int32_t bufId = this->template GetBufId<pos>(pingPong);
        LocalTensor<inputType> inTensor = this->tensorPool_[bufId * this->blockLen_]
                                              .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inTensor.SetBufferLen(this->blockEleNum_);
#endif
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(this->inGm_[input::Pos] + gmOffset * sizeof(inputType)));
        // Set getBuf
        GetTensor<TPosition::VECIN>(bufId);

        // Run copyIn
        int64_t block_count = 1;
        int64_t src_stride = 0;
        int64_t dst_stride = 0;
        int64_t block_len = handleLength;
        uint32_t loop1Size = 1;
        uint64_t loop1SrcStride = 0;
        uint64_t loop1DstStride = 0;
        uint32_t loop2Size = 1;
        uint64_t loop2SrcStride = 0;
        uint64_t loop2DstStride = 0;

        if (this->runningRank_ > 1) {
            this->template GetStridesWithPadThroughDims(tilingData->inputDims[input::Pos],
                                                        inputStridesWithPad[input::Pos]);
            block_count = handleLength;
            block_len = tilingData->inputDims[input::Pos][tilingData->shapeLen - 1];
            src_stride = tilingData->inputStrides[input::Pos][tilingData->shapeLen - NUM_TWO] - block_len;

            int64_t inputTypeAlignSize = blockSize / sizeof(inputType);
            int64_t blockLenInputTypeAlignNum = (block_len + inputTypeAlignSize - 1) / inputTypeAlignSize *
                                                inputTypeAlignSize;
            dst_stride = ((inputStridesWithPad[input::Pos][tilingData->shapeLen - NUM_TWO] -
                           blockLenInputTypeAlignNum) *
                          sizeof(inputType)) /
                         blockSize;
        }

        if (this->runningRank_ > NUM_TWO) {
            block_count = tilingData->inputDims[input::Pos][tilingData->shapeLen - NUM_TWO];
            loop1Size = handleLength;
            loop1SrcStride = tilingData->inputStrides[input::Pos][tilingData->shapeLen - NUM_THREE] * sizeof(inputType);
            loop1DstStride = inputStridesWithPad[input::Pos][tilingData->shapeLen - NUM_THREE] * sizeof(inputType);
        }

        if (this->runningRank_ > NUM_THREE) {
            loop1Size = tilingData->inputDims[input::Pos][tilingData->shapeLen - NUM_THREE];
            loop2Size = handleLength;
            loop2SrcStride = tilingData->inputStrides[input::Pos][tilingData->shapeLen - NUM_FOUR] * sizeof(inputType);
            loop2DstStride = inputStridesWithPad[input::Pos][tilingData->shapeLen - NUM_FOUR] * sizeof(inputType);
        }

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(block_count), static_cast<uint32_t>(block_len * sizeof(inputType)),
            static_cast<uint32_t>(src_stride * sizeof(inputType)), static_cast<uint32_t>(dst_stride), 0};
        AscendC::DataCopyPadExtParams<inputType> padParams{false, 0, 0, 0};

        AscendC::LoopModeParams LoopParams{loop1Size,      loop2Size,      loop1SrcStride,
                                           loop1DstStride, loop2SrcStride, loop2DstStride};
        AscendC::SetLoopModePara(LoopParams, DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<inputType>(inTensor, globalTensor, copyParams, padParams);
        AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        // Set rlsBuf
        ReleaseTensor<TPosition::VECIN>(bufId);
    }

    template <class Op, int start = 0>
    __aicore__ constexpr static inline int GetFunOutputPosition()
    {
        if constexpr (std::is_same<typename BrcDag::FunList::template At<start>, Op>::value) {
            return start;
        } else if constexpr (start + 1 < BrcDag::FunList::Size) {
            return GetFunOutputPosition<Op, start + 1>();
        } else {
            static_assert(start + 1 < BrcDag::FunList::Size, "The required output in FunList is not found.");
            return -1;
        }
    }

    /* *
     * 输出Op
     * @tparam Op CopyOut的Bind节点
     * @tparam pos CopyOut在FunList内的索引
     * @param offset 当前处理的数据在GM上的偏移
     * @param tileLength 当前处理的数据块大小
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return void
     */
    template <typename Op, int pos>
    __aicore__ inline void CopyOutMoveAlign(int64_t handleLength, uint64_t offset, int32_t pingPong)
    {
        static_assert(Op::Args::Size == 2, "Input args should be 2");
        using input = typename Op::Args::template At<1>;
        using output = typename Op::Args::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        static_assert(Placeholder::IsOutHolder<output>::Value, "output args should be out holder");
        if constexpr (std::is_same<typename output::DType, uint1_t>::value) {
            static_assert(std::is_same<inputType, uint8_t>::value,
                          "CopyOut data type is inconsistent with out holder data type.");
            offset = offset / BYTE_LENGTH;
        } else {
            static_assert(std::is_same<typename output::DType, inputType>::value,
                          "CopyOut data type is inconsistent with Op data type.");
        }

        // Prepare input args
        int32_t bufId = this->template GetBufId<GetFunOutputPosition<input>()>(pingPong);
        LocalTensor<inputType> localTensor = this->tensorPool_[bufId * this->blockLen_]
                                                 .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        localTensor.SetBufferLen(this->blockEleNum_);
#endif
        static_assert(output::Pos < BrcDag::OutputSize, "output Pos is not less than output number.");
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(this->outGm_[output::Pos] + offset * sizeof(inputType)));
        // Set getBuf
        GetTensor<TPosition::VECOUT>(bufId);
        // Run func
        int64_t block_count = 1;
        int64_t src_stride = 0;
        int64_t block_len = handleLength;
        uint32_t loop1Size = 1;
        uint64_t loop1SrcStride = 0;
        uint64_t loop1DstStride = 0;
        uint32_t loop2Size = 1;
        uint64_t loop2SrcStride = 0;
        uint64_t loop2DstStride = 0;

        if (this->runningRank_ > 1) {
            block_count = handleLength;
            block_len = tilingData->outputDims[tilingData->shapeLen - 1];

            int64_t outputTypeAlignSize = blockSize / sizeof(inputType);
            int64_t blockLenOutputTypeAlignNum = (block_len + outputTypeAlignSize - 1) / outputTypeAlignSize *
                                                 outputTypeAlignSize;
            src_stride = ((tilingData->outputStridesWithPad[tilingData->shapeLen - NUM_TWO] -
                           blockLenOutputTypeAlignNum) *
                          sizeof(inputType)) /
                         blockSize;
        }

        if (this->runningRank_ > NUM_TWO) {
            block_count = tilingData->outputDims[tilingData->shapeLen - NUM_TWO];
            loop1Size = handleLength;
            loop1SrcStride = tilingData->outputStridesWithPad[tilingData->shapeLen - NUM_THREE] * sizeof(inputType);
            loop1DstStride = tilingData->outputStrides[tilingData->shapeLen - NUM_THREE] * sizeof(inputType);
        }

        if (this->runningRank_ > NUM_THREE) {
            loop1Size = tilingData->outputDims[tilingData->shapeLen - NUM_THREE];
            loop2Size = handleLength;
            loop2SrcStride = tilingData->outputStridesWithPad[tilingData->shapeLen - NUM_FOUR] * sizeof(inputType);
            loop2DstStride = tilingData->outputStrides[tilingData->shapeLen - NUM_FOUR] * sizeof(inputType);
        }

        DataCopyExtParams copyParams{static_cast<uint16_t>(block_count),
                                     static_cast<uint32_t>(block_len * sizeof(inputType)),
                                     static_cast<uint32_t>(src_stride), 0, 0};

        AscendC::LoopModeParams LoopParams{loop1Size,      loop2Size,      loop1SrcStride,
                                           loop1DstStride, loop2SrcStride, loop2DstStride};
        AscendC::SetLoopModePara(LoopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad<inputType>(globalTensor, localTensor, copyParams);
        AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

        // Set rlsBuf
        ReleaseTensor<TPosition::VECOUT>(bufId);
    }

    // 遍历执行图
    template <int pos = 0, bool insideIf = false>
    __aicore__ inline void Run(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM],
                               int64_t ubLoopIdx, int32_t pingPong)
    {
        if constexpr (pos >= BrcDag::FunList::Size) {
            return;
        }

        // Run current func
        using Op = typename BrcDag::FunList::template At<pos>;
        using Func = typename Op::Fun;
        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value && !insideIf) {
            constexpr int vecIndex = BrcDag::template VecBrcIdxDepend<Op>;
            if constexpr (vecIndex >= 0) {
                if ((tilingData->inputVecBrcStrides[vecIndex] != 0) ||
                    (ubLoopIdx == 0 ||
                     (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter == 0)) {
                    Run<pos, true>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                }
                using vecBrcOp = typename BrcDag::VecBrcNodes::template At<vecIndex>;
                constexpr int vecBrcOpPos = BrcDag::FunList::template GetIndex<vecBrcOp>();
                Run<vecBrcOpPos + 1, false>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                return;
            }
        }

        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
            this->template CopyInMoveAlign<Op, pos>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyInBrc>::Value) {
            CopyInAndVecBrc<Op, pos>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
            int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->outputStrides, tilingData->ubSplitAxis,
                                                    tilingData->ubFormer);
            // int64_t tileLength = ubSplitSize * tilingData->outputStrides[tilingData->ubSplitAxis];
            this->template CopyOutMoveAlign<Op, pos>(ubSplitSize, gmOffset, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            this->template VecBroadcast<Op, pos>(ubLoopIdx, pingPong);
        } else {
            uint64_t tileLength = ubSplitSize *
                                  tilingData->outputStridesWithPad[tilingData->ubSplitAxis]; /// 長度是补pad之后的
            this->template RunNormalOp<Op, pos>(tileLength, pingPong);
        }

        if constexpr (insideIf && __aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            return;
        }

        // Run next func
        if constexpr (pos + 1 < BrcDag::FunList::Size) {
            Run<pos + 1, insideIf>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        }
    }

private:
    constexpr static int bufferNum = BrcDag::template GetBufferNum<false, true>();
    const BroadcastNlastTransposeTilingData<BrcDag>* tilingData;

    constexpr static uint32_t COPY_BRC_SIZE = BrcDag::CopyBrcSize > 0 ? BrcDag::CopyBrcSize : 1;
    uint32_t copyInBrcFormerShape[COPY_BRC_SIZE][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {{1}};
    uint32_t copyInBrcFormerShapeWithLastPad[COPY_BRC_SIZE][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {{1}};
    uint32_t inputBrcStridesWithPad[COPY_BRC_SIZE][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {{1}};
    uint32_t inputStridesWithPad[BrcDag::InputSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM] = {{1}};

    uint32_t cBrcFormer[COPY_BRC_SIZE] = {1};
    uint32_t cBrcTail[COPY_BRC_SIZE] = {1};

    uint32_t cBrcIndex_ = 0;
    int64_t ubLoopNum_ = 0;
    uint32_t blockSize = Ops::Base::GetUbBlockSize();
};
} // namespace Base
} // namespace Ops

#endif // BROADCAST_SCH_NLAST_TRANSPOSE_UB_BROADCAST_H_