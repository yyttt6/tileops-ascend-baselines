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
 * \file broadcast_sch_base_transpose.h
 * \brief
 */
#ifndef BROADCAST_SCH_BASE_TRANSPOSE_H_
#define BROADCAST_SCH_BASE_TRANSPOSE_H_

#include "atvoss/util/kernel_vec_intf_apt.h"
#include "adv_api/pad/broadcast.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/dfx.h"
#include "atvoss/util/sync.h"
#include "broadcast_utils_transpose.h"
#include "atvoss/broadcast/broadcast_base_struct.h"
namespace Ops {
namespace Base {
static constexpr uint64_t NON_CONTIGUOUS_BYTE_LENGTH = 8;

template <class BrcDag, bool UseNddma, int64_t R = -1>
class BroadcastBaseNonContiguousSch {
public:
    __aicore__ inline explicit BroadcastBaseNonContiguousSch(const BroadcastBaseTilingData<BrcDag>* baseTilingData)
        : tilingData_(baseTilingData)
    {}

protected:
    /* *
     * 获取UBBRC的shape信息
     * @tparam N shape的长度
     * @param oriShape 原始shape
     * @param inputUblength 输入UB长度
     * @param ubFormershape UB主块shape
     * @return void
     */
    template <size_t N>
    __aicore__ inline void GetUbBroadcastShapeInfo(const int64_t (&oriShape)[N], int64_t inputUblength[2],
                                                   uint32_t (&ubFormershape)[N])
    {
        int64_t ubFormerLength = 1;
        int64_t ubTailLength = 1;
        if (oriShape[tilingData_->ubSplitAxis] == 1) {
            ubFormershape[0] = 1;
        } else {
            ubTailLength = tilingData_->ubTail;
            ubFormershape[0] = tilingData_->ubFormer;
            ubFormerLength = tilingData_->ubFormer;
        }

        int j = 1;
        for (uint64_t i = tilingData_->ubSplitAxis + 1; i < tilingData_->shapeLen; i++) {
            ubFormershape[j] = oriShape[i];
            ubFormerLength *= oriShape[i];
            ubTailLength *= oriShape[i];
            j = j + 1;
        }

        inputUblength[0] = ubFormerLength;
        inputUblength[1] = ubTailLength;
    }

    /* *
     * 获取vecBrc信息
     * @tparam index 当前处理的索引
     * @return void
     */
    template <int index = 0>
    __aicore__ inline void GetVecBrcInfo()
    {
        using vecBrcOp = typename BrcDag::VecBrcNodes::template At<index>;
        using inputType = typename vecBrcOp::template FunInArgType<0>;
        int64_t vecBrcInputUbLength[2] = {1, 1};
        GetUbBroadcastShapeInfo(tilingData_->inputVecBrcDims[index], vecBrcInputUbLength, vecBrcFormerShape_[index]);
        if (tilingData_->inputVecBrcDims[index][tilingData_->ubSplitAxis] != 1) {
            vBrcFormer_[index] = tilingData_->ubFormer;
            vBrcTail_[index] = tilingData_->ubTail;
        } else {
            vBrcFormer_[index] = 1;
            vBrcTail_[index] = 1;
        }
        if constexpr (index + 1 < BrcDag::VecBrcSize) {
            GetVecBrcInfo<index + 1>();
        }
    }

    /* *
     * VecBrc节点调用的Broadcast函数
     * @tparam T 数据类型
     * @param dst 目的tensor
     * @param src 源tensor
     * @param realIdx 当前处理的索引
     * @return void
     */
    template <typename T>
    __aicore__ inline void VecBrc(LocalTensor<T>& dst, LocalTensor<T>& src, int64_t realIdx)
    {
        BroadcastTiling runningTiling;
        if (realIdx == tilingData_->ubOuter - 1) {
            dstUbFormerShape_[0] = ubTail_;
            vecBrcFormerShape_[vBrcIndex_][0] = vBrcTail_[vBrcIndex_];
        } else {
            dstUbFormerShape_[0] = ubFormer_;
            vecBrcFormerShape_[vBrcIndex_][0] = vBrcFormer_[vBrcIndex_];
        }
        if constexpr (R == -1) {
            GetBroadcastTilingInfo<T>(runningRank_, dstUbFormerShape_, vecBrcFormerShape_[vBrcIndex_], false,
                                      runningTiling);
            Broadcast(dst, src, dstUbFormerShape_, vecBrcFormerShape_[vBrcIndex_], &runningTiling);
        } else {
            GetBroadcastTilingInfo<T, R>(R, dstUbFormerShape_, vecBrcFormerShape_[vBrcIndex_], false, runningTiling);
            Broadcast<T, R>(dst, src, dstUbFormerShape_, vecBrcFormerShape_[vBrcIndex_], &runningTiling);
        }
    }

    /* *
     * VecBroadcast函数
     * @tparam Op VecBrc的Bind节点
     * @tparam pos VecBrc在FunList内的索引
     * @param ubLoopIdx 当前处理的循环变量
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return void
     */
    template <typename Op, int pos>
    __aicore__ inline void VecBroadcast(int64_t ubLoopIdx, int32_t pingPong)
    {
        static_assert(Op::Args::Size == 1, "Broadcast input args should be 1.");

        // get current broadcast op tensor
        using outputType = typename Op::template FunRetArgType<0>;
        int32_t bufId = GetBufId<pos>(pingPong);
        LocalTensor<outputType> outTensor = tensorPool_[bufId * blockLen_].template ReinterpretCast<outputType>();
#ifndef __CCE_KT_TEST__
        outTensor.SetBufferLen(blockEleNum_);
#endif
        GetTensor<TPosition::VECCALC>(bufId);
        // get current broadcast op input tensor
        using inputOp = typename Op::Args::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;

        int32_t inputBufId = GetBufId<GetFunOutputPos<inputOp>()>(pingPong);
        LocalTensor<inputType> inputTensor = tensorPool_[inputBufId * blockLen_].template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inputTensor.SetBufferLen(blockEleNum_);
#endif
        int64_t realIdx = (AscendC::GetBlockIdx() * tilingData_->blockFormer + ubLoopIdx) % tilingData_->ubOuter;
        static_assert(std::is_same<inputType, outputType>::value,
                      "Broadcast inputType  is inconsistent with outputType.");
        VecBrc<inputType, true>(outTensor, inputTensor, realIdx);
        ReleaseTensor<TPosition::VECCALC>(inputBufId);
        ReleaseTensor<TPosition::VECCALC>(bufId);
        vBrcIndex_++;
    }

    /* *
     * SetScalar函数
     * @tparam idx 当前处理的索引
     * @param offset 当前处理的偏移量
     * @return void
     */
    template <int idx = 0>
    __aicore__ inline void SetScalar(int offset)
    {
        if constexpr (idx < BrcDag::VarSize) {
            using VarPlaceHolder = typename BrcDag::Vars::template At<idx>;
            using DType = typename VarPlaceHolder::DType;
            scalars_.template Set<idx>(reinterpret_cast<const DType&>(tilingData_->scalarData[offset]));

            if constexpr (idx + 1 < BrcDag::VarSize) {
                offset += sizeof(DType);
                SetScalar<idx + 1>(offset);
            }
        }
    }

    /* *
     * GetBufIdNddma函数
     * @tparam pos 在FunList内的索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return 缓冲区ID
     */
    template <int pos>
    __aicore__ inline constexpr auto GetBufIdNddma(int32_t pingPong)
    {
        constexpr int32_t bufIdPing = cacheBufferIds_[0][pos];
        constexpr int32_t bufIdPong = cacheBufferIds_[1][pos];

        if constexpr (bufIdPing == bufIdPong) {
            return bufIdPing;
        } else {
            return pingPong == 0 ? bufIdPing : bufIdPong;
        }
    }

    /* *
     * GetBufIdUbBrc函数
     * @tparam pos 在FunList内的索引
     * @tparam isList 是否为列表
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return 缓冲区ID
     */
    template <int pos, bool isList = false>
    __aicore__ inline constexpr auto GetBufIdUbBrc(int32_t pingPong)
    {
        constexpr int32_t bufIdPing = cacheBufferIds_[0][pos];
        constexpr int32_t bufIdPong = cacheBufferIds_[1][pos];
        constexpr auto decodeBufferIdPing = DecodeBufferId<bufIdPing>::Value;
        constexpr auto decodeBufferIdPong = DecodeBufferId<bufIdPong>::Value;
        if constexpr (isList) {
            static_assert(CombinedBufferCount<bufIdPing>::Value >= 2, "CombinedBufferCount should greater than 2.");
            static_assert(CombinedBufferCount<bufIdPong>::Value >= 2, "CombinedBufferCount should greater than 2.");
            if constexpr (bufIdPing == bufIdPong) {
                return decodeBufferIdPing;
            } else {
                return pingPong == 0 ? decodeBufferIdPing : decodeBufferIdPong;
            }
        } else {
            static_assert(CombinedBufferCount<bufIdPing>::Value >= 1, "CombinedBufferCount should greater than 1.");
            static_assert(CombinedBufferCount<bufIdPong>::Value >= 1, "CombinedBufferCount should greater than 1.");
            if constexpr (bufIdPing == bufIdPong) {
                return decodeBufferIdPing[0];
            } else {
                return pingPong == 0 ? decodeBufferIdPing[0] : decodeBufferIdPong[0];
            }
        }
    }

    /* *
     * GetBufId函数
     * @tparam pos 在FunList内的索引
     * @tparam isList 是否为列表
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return 缓冲区ID
     */
    template <int pos, bool isList = false>
    __aicore__ inline constexpr auto GetBufId(int32_t pingPong)
    {
        if constexpr (UseNddma) {
            return GetBufIdNddma<pos>(pingPong);
        } else {
            return GetBufIdUbBrc<pos, isList>(pingPong);
        }
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
    // 现存问题，多个CopyIn节点还没考虑
    template <typename Op, int pos>
    __aicore__ inline void CopyIn(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                  int64_t ubLoopIdx, int32_t pingPong, bool withLoop)
    {
        static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");
        using input = typename Op::InHolders::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        if constexpr (Op::IsScalarOp) {
            inputType scalar = GetScalar<inputType, input>();
            opScalars_.template Set<pos>(scalar);
            return;
        }
        // Prepare input args
        int32_t bufId = GetBufId<pos>(pingPong);
        LocalTensor<inputType> inTensor = tensorPool_[bufId * blockLen_].template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inTensor.SetBufferLen(blockEleNum_);
#endif
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ inputType*>(inGm_[input::Pos]));
        // Set getBuf
        GetTensor<TPosition::VECIN>(bufId);
        // Run NDDMA
        if (withLoop) {
            BroadcastNddmaNonContiguousWithLoop(globalTensor, inTensor, tilingData_->outputDims,
                                                tilingData_->outputStrides, tilingData_->inputStrides[input::Pos],
                                                axesIndices, tilingData_->ubSplitAxis, tilingData_->shapeLen,
                                                ubSplitSize, tilingData_->ubFormer);
        } else {
            BroadcastNddmaNonContiguousWithoutLoop(globalTensor, inTensor, tilingData_->outputDims,
                                                   tilingData_->outputStrides, tilingData_->inputStrides[input::Pos],
                                                   axesIndices, tilingData_->ubSplitAxis, tilingData_->shapeLen,
                                                   ubSplitSize, tilingData_->ubFormer);
        }
        // Set rlsBuf
        ReleaseTensor<TPosition::VECIN>(bufId);
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
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t tileLength, int32_t pingPong)
    {
        static_assert(Op::Args::Size == 2, "Input args should be 2");
        using input = typename Op::Args::template At<1>;
        using output = typename Op::Args::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        static_assert(Placeholder::IsOutHolder<output>::Value, "output args should be out holder");
        if constexpr (std::is_same<typename output::DType, uint1_t>::value) {
            static_assert(std::is_same<inputType, uint8_t>::value,
                          "CopyOut data type is inconsistent with out holder data type.");
            offset = offset / NON_CONTIGUOUS_BYTE_LENGTH;
            tileLength = tileLength / NON_CONTIGUOUS_BYTE_LENGTH;
        } else {
            static_assert(std::is_same<typename output::DType, inputType>::value,
                          "CopyOut data type is inconsistent with Op data type.");
        }

        // Prepare input args
        int32_t bufId = GetBufId<GetFunOutputPos<input>()>(pingPong);
        LocalTensor<inputType> localTensor = tensorPool_[bufId * blockLen_].template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        localTensor.SetBufferLen(blockEleNum_);
#endif
        static_assert(output::Pos < BrcDag::OutputSize, "output Pos is not less than output number.");
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(outGm_[output::Pos] + offset * sizeof(inputType)));
        // Set getBuf
        GetTensor<TPosition::VECOUT>(bufId);
        // Run func
        Vec::CopyOut<inputType>(globalTensor, localTensor, tileLength);
        // Set rlsBuf
        ReleaseTensor<TPosition::VECOUT>(bufId);
    }

    /* *
     * 遍历DAG，计算Op的position
     * @tparam Op 待计算的Op, 需要是Bind类型
     * @tparam start 查询起始位置的Position，默认从FunList的第0个开始匹配
     * @return Op在FunList中的索引坐标
     */
    template <class Op, int start = 0>
    __aicore__ constexpr static inline int GetFunOutputPos()
    {
        if constexpr (std::is_same<typename BrcDag::FunList::template At<start>, Op>::value) {
            return start;
        } else if constexpr (start + 1 < BrcDag::FunList::Size) {
            return GetFunOutputPos<Op, start + 1>();
        } else {
            static_assert(start + 1 < BrcDag::FunList::Size, "The required output in FunList is not found.");
            return -1;
        }
    }

    /* *
     * 根据DAG中Scalar描述获取Scalar的值
     * @tparam ScalarType Scalar的数据类型
     * @tparam scalarValue 存储Scalar值的Holder，主要有Var/InHolder/ConstValue三种类型的Scalar
     * @return 类型位ScalarType的Scalar值
     */
    template <typename ScalarType, typename scalarValue>
    __aicore__ inline constexpr ScalarType GetScalar()
    {
        static_assert(!(Placeholder::IsVar<scalarValue>::Value && Placeholder::IsInHolder<scalarValue>::Value &&
                        Placeholder::IsConstValue<scalarValue>::Value),
                      "The input parameter type is not FunBind, Var, Const or Holder.");
        if constexpr (Placeholder::IsVar<scalarValue>::Value) {
            ScalarType scalar = scalars_.template Get<scalarValue::Pos>();
            return scalar;
        } else if constexpr (Placeholder::IsInHolder<scalarValue>::Value) {
            GlobalTensor<ScalarType> globalTensor;
            globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalarType*>(inGm_[scalarValue::Pos]));
            ScalarType scalar = globalTensor.GetValue(0);
            return scalar;
        } else if constexpr (Placeholder::IsConstValue<scalarValue>::Value) {
            ScalarType scalar = static_cast<ScalarType>(scalarValue::value);
            return scalar;
        }
    }

    template <typename Op, int argPos>
    __aicore__ inline auto ConvertArgs(int32_t pingPong)
    {
        using InputOp = typename Op::InArgs::template At<argPos>;
        using TensorType = typename Op::template FunInArgType<argPos>;
        if constexpr (__aux::TypeIsFunBind<InputOp>::Value) {
            if constexpr (InputOp::IsScalarOp) {
                TensorType scalar = opScalars_.template Get<GetFunOutputPos<InputOp>()>();
                return scalar;
            } else {
                // 判断当前输入是否是同一个操作的多引用输出，避免重复插入同步
                constexpr static bool isDuplicate = Op::InArgs::template IsExist<InputOp, argPos + 1>();
                int32_t bufId = GetBufId<GetFunOutputPos<InputOp>()>(pingPong);
                if constexpr (!isDuplicate) {
                    GetTensor<TPosition::VECCALC>(bufId);
                }
                LocalTensor<TensorType> inputTensor = tensorPool_[bufId * blockLen_]
                                                          .template ReinterpretCast<TensorType>();
#ifndef __CCE_KT_TEST__
                inputTensor.SetBufferLen(blockEleNum_);
#endif
                return inputTensor;
            }
        } else {
            TensorType scalar = GetScalar<TensorType, InputOp>();
            return scalar;
        }
    }

    template <typename Op, int argPos>
    __aicore__ inline void TryReleaseArgs(int32_t pingPong)
    {
        using InputOp = typename Op::InArgs::template At<argPos>;
        if constexpr (__aux::TypeIsFunBind<InputOp>::Value) {
            if constexpr (!InputOp::IsScalarOp) {
                // 判断当前输入是否是同一个操作的多引用输出，避免重复插入同步
                constexpr static bool isDuplicate = Op::InArgs::template IsExist<InputOp, argPos + 1>();
                int32_t bufId = GetBufId<GetFunOutputPos<InputOp>()>(pingPong);
                if constexpr (!isDuplicate) {
                    ReleaseTensor<TPosition::VECCALC>(bufId);
                }
            }
        }
    }

    template <typename Op, size_t... I>
    __aicore__ inline auto MakeArgs(int32_t pingPong, AscendC::Std::index_sequence<I...>)
    {
        return AscendC::Std::make_tuple(ConvertArgs<Op, I>(pingPong)...);
    }

    template <typename Op>
    __aicore__ inline auto PrepareArgs(int32_t pingPong)
    {
        return MakeArgs<Op>(pingPong, AscendC::Std::make_index_sequence<Op::InputSize>{});
    }

    template <typename Func, typename OutputType, typename Tuple, size_t... I>
    __aicore__ inline auto CallImpl(LocalTensor<OutputType>& outTensor, Tuple& inputs, uint64_t tileLength,
                                    AscendC::Std::index_sequence<I...>)
    {
        return Func(outTensor, AscendC::Std::get<I>(inputs)..., tileLength);
    }

    template <typename Func, typename OutputType, typename Tuple>
    __aicore__ inline auto Call(LocalTensor<OutputType>& outTensor, Tuple& inputs, uint64_t tileLength)
    {
        return CallImpl<Func, OutputType>(outTensor, inputs, tileLength,
                                          AscendC::Std::make_index_sequence<AscendC::Std::tuple_size<Tuple>::value>{});
    }

    template <typename Func, typename OutputType, typename Tuple, size_t... I>
    __aicore__ inline auto CallImpl(OutputType& outScalar, Tuple& inputs, uint64_t tileLength,
                                    AscendC::Std::index_sequence<I...>)
    {
        return Func(outScalar, AscendC::Std::get<I>(inputs)..., tileLength);
    }

    template <typename Func, typename OutputType, typename Tuple>
    __aicore__ inline auto Call(OutputType& outScalar, Tuple& inputs, uint64_t tileLength)
    {
        return CallImpl<Func, OutputType>(outScalar, inputs, tileLength,
                                          AscendC::Std::make_index_sequence<AscendC::Std::tuple_size<Tuple>::value>{});
    }

    template <typename Op, size_t... I>
    __aicore__ inline void DoPostArgs(int32_t pingPong, AscendC::Std::index_sequence<I...>)
    {
        (TryReleaseArgs<Op, I>(pingPong), ...);
    }

    template <typename Op>
    __aicore__ inline void PostArgs(int32_t pingPong)
    {
        DoPostArgs<Op>(pingPong, AscendC::Std::make_index_sequence<Op::InputSize>{});
    }

    /* *
     * 执行Op
     * @tparam OutputType 输出数据类型
     * @tparam Op 当前计算节点，需要是Bind类型
     * @tparam pos 当前计算节点的位置
     * @param outTensor 输出的LocalTensor对象
     * @param tileLength 单次处理的数据个数
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return
     */
    template <typename OutputType, class Op, int pos = 0>
    __aicore__ inline constexpr void RunOp(LocalTensor<OutputType>& outTensor, uint64_t tileLength, int32_t pingPong)
    {
        using Func = typename Op::Fun;
        auto inputArgs = PrepareArgs<Op>(pingPong);
        Call<Func, OutputType>(outTensor, inputArgs, tileLength);
        PostArgs<Op>(pingPong);
    }

    /* *
     * 执行常规计算节点
     * @tparam Op 除CopyIn/CopyOut之外的常规计算节点，Bind类型
     * @tparam pos 当前计算节点在FunList中的索引
     * @param tileLength 当前计算的数据块大小
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return
     */
    template <class Op, int pos = 0>
    __aicore__ inline constexpr void RunNormalOp(uint64_t tileLength, int32_t pingPong)
    {
        // Prepare output args
        using OutputType = typename Op::template FunRetArgType<0>;

        int32_t bufId = GetBufId<pos>(pingPong);
        LocalTensor<OutputType> outTensor = tensorPool_[bufId * blockLen_].template ReinterpretCast<OutputType>();
#ifndef __CCE_KT_TEST__
        outTensor.SetBufferLen(blockEleNum_);
#endif
        // Set getBuf
        GetTensor<TPosition::VECCALC>(bufId);
        // Run Op
        RunOp<OutputType, Op, pos>(outTensor, tileLength, pingPong);

        // Set rlsBuf
        ReleaseTensor<TPosition::VECCALC>(bufId);
    }

protected:
    int blockLen_ = 0;
    int blockEleNum_ = 0;
    uint32_t runningRank_ = 0;
    uint32_t ubFormer_ = 1;
    uint32_t ubTail_ = 1;
    uint32_t vBrcIndex_ = 0;

    GM_ADDR inGm_[BrcDag::InputSize];
    GM_ADDR outGm_[BrcDag::OutputSize];
    LocalTensor<uint8_t> tensorPool_;
    typename BrcDag::VarType scalars_;
    typename BrcDag::ScalarOpType opScalars_;

    constexpr static uint32_t VEC_BRC_SIZE = BrcDag::VecBrcSize > 0 ? BrcDag::VecBrcSize : 1;
    uint32_t vecBrcFormerShape_[VEC_BRC_SIZE][8] = {{1}};
    uint32_t vBrcFormer_[VEC_BRC_SIZE] = {1};
    uint32_t vBrcTail_[VEC_BRC_SIZE] = {1};
    uint32_t dstUbFormerShape_[8] = {1};

    constexpr static auto cacheBufferIds_ = BrcDag::template GetBufferIds<UseNddma, true>();

private:
    // const BroadcastDiscontiguousTilingData<BrcDag> *tilingData_;
    const BroadcastBaseTilingData<BrcDag>* tilingData_;
};
} // namespace Base
} // namespace Ops

#endif // BROADCAST_SCH_BASE_TRANSPOSE_H_