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
 * \file elewise_sch_16b.h
 * \brief
 */
#ifndef ELEWISE_SCH_16B_H_
#define ELEWISE_SCH_16B_H_

#include "atvoss/util/dag.h"
#include "atvoss/util/kernel_vec_intf_apt.h"
#include "atvoss/util/sync.h"
#include "atvoss/util/vec.h"
#include "elewise_base_struct.h"

#pragma "lib"

namespace Ops {
namespace Base {

template <uint64_t schMode, class ElemDag>
class ElementwiseSch16B {
public:
    __aicore__ inline explicit ElementwiseSch16B(__tiling_data_ptr__ EleBaseTilingData16B* baseTilingData)
    {
        AscendC::InitSocState();
        tilingData = baseTilingData;

        dim0_ = tilingData->dim0;
        ubFormer_ = tilingData->ubFormer;
        coreNum_ = tilingData->coreNum;

        blockFormer_ = ((dim0_ + coreNum_ - 1) / coreNum_ + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH *
                       CACHE_LINE_BYTE_LENGTH;
        blockNum_ = (dim0_ + blockFormer_ - 1) / blockFormer_;
        blockTail_ = dim0_ - (blockNum_ - 1) * blockFormer_;
        ubLoopOfFormerBlock_ = (blockFormer_ + ubFormer_ - 1) / ubFormer_;
        ubLoopOfTailBlock_ = (blockTail_ + ubFormer_ - 1) / ubFormer_;
        ubTailOfFormerBlock_ = blockFormer_ - (ubLoopOfFormerBlock_ - 1) * ubFormer_;
        ubTailOfTailBlock_ = blockTail_ - (ubLoopOfTailBlock_ - 1) * ubFormer_;
    }

    /**
     * 初始化ElementwiseSch16B对象
     * @tparam Args 输入Args类型
     * @param args 输入args参数列表，需要匹配DAG图中PlaceHolder的顺序[In0, In1..., Out0, Out1...]
     */
    template <class... Args>
    __aicore__ inline void Init(Args... args)
    {
        static_assert(inputNums + outputNums == sizeof...(Args),
                      "ElementwiseSch16B.Init args num should match DAG holders.");
        InitInputArgs<0>(args...); // 调用入参分析,input,output
        RUN_LOG("BufferNum: %d, Mte2Num: %d, Mte3Num: %d, BufLevel: %d", ElemDag::BufferNum, ElemDag::Mte2Num,
                ElemDag::Mte3Num, ElemDag::BufLevel);
        blockLen = ubFormer_ * ElemDag::MaxDtypeBytes;
    }

    /**
     * 执行ElementwiseSch16B
     */
    __aicore__ inline void Process()
    {
        uint64_t offset = 0;
        uint64_t loopNum = GetBlockIdx() == blockNum_ - 1 ? ubLoopOfTailBlock_ : ubLoopOfFormerBlock_;
        uint64_t tailNum = GetBlockIdx() == blockNum_ - 1 ? ubTailOfTailBlock_ : ubTailOfFormerBlock_;
        uint64_t i = 0;
        for (; i < loopNum - 1; i++) {
            // ub整循环处理的元素个数
            Run<0>(offset, ubFormer_, i & 1);
            offset += ubFormer_;
        }

        // ub尾循环处理的元素个数
        Run<0>(offset, tailNum, i & 1);
    }

    /**
     * 设置DAG图上Var类型PlaceHolder的值
     * @tparam U Var的数据类型
     * @tparam index Var的索引
     * @param value
     */
    template <typename U, int index>
    __aicore__ inline void SetVar(U value)
    {
        static_assert(index < ElemDag::Vars::Size, "The index exceeds the number of Vars defined in DAG.");
        scalars.template Set<index>(value);
    }

protected:
    /**
     * 当前核处理数据的起始偏移
     * @tparam DataType 当前输入的数据类型，当数据类型是1bit时，偏移需要除以8
     * @return
     */
    template <typename DataType>
    __aicore__ inline int64_t CalcBlockOffset()
    {
        if constexpr (std::is_same<DataType, uint1_t>::value) {
            return blockFormer_ * GetBlockIdx() / BYTE_LENGTH;
        }
        return blockFormer_ * GetBlockIdx() * sizeof(DataType);
    }

    template <int target, typename T>
    struct GetHolderByPos {
        using Type = void;
    };

    template <int target, template <typename...> typename Holders, typename Holder>
    struct GetHolderByPos<target, Holders<Holder>> {
        using Type = __aux::Condition<target == Holder::Pos, Holder, void>;
    };

    template <int target, template <typename...> typename Holders, typename Holder, typename... HolderTs>
    struct GetHolderByPos<target, Holders<Holder, HolderTs...>> {
        using Type = __aux::Condition<target == Holder::Pos, Holder,
                                      typename GetHolderByPos<target, Holders<HolderTs...>>::Type>;
    };

    // 初始化输出
    template <int start, class... Args>
    __aicore__ inline void InitOutputArgs(GM_ADDR y, Args... args)
    {
        if constexpr (start < outputNums) {
            using Holder = typename GetHolderByPos<start, typename ElemDag::OutHolders>::Type;
            using DataType = typename Holder::DType;
            outGm[start].SetGlobalBuffer((__gm__ uint8_t*)y + CalcBlockOffset<DataType>());
        }

        if constexpr (start + 1 < outputNums) {
            InitOutputArgs<start + 1>(args...);
        }
    }

    // 初始化输入
    template <int start, class... Args>
    __aicore__ inline void InitInputArgs(GM_ADDR x, Args... args)
    {
        if constexpr (start < inputNums) {
            using Holder = typename GetHolderByPos<start, typename ElemDag::InHolders>::Type;
            using DataType = typename Holder::DType;
            if constexpr (Placeholder::IsInScalar<Holder>::Value) {
                inGm[start].SetGlobalBuffer((__gm__ uint8_t*)x);
            } else {
                inGm[start].SetGlobalBuffer((__gm__ uint8_t*)x + CalcBlockOffset<DataType>());
            }
        }

        if constexpr (start + 1 <= inputNums) {
            InitInputArgs<start + 1>(args...);
        } else {
            // inputNums == 0时，x实际时第一个输出
            InitOutputArgs<0>(x, args...);
        }
    }

    /**
     * 搬入Op
     * @tparam Op CopyIn的Bind节点
     * @tparam pos CopyIn在FunList内的索引
     * @param offset 当前处理的数据在GM上的偏移
     * @param tileLength 当前处理的数据块大小
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename Op, int pos>
    __aicore__ inline void CopyIn(uint64_t offset, uint64_t tileLength, int32_t pingPong)
    {
        static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");

        using InputOp = typename Op::InHolders::template At<0>;
        using TensorType = typename Op::template FunInArgType<0>;
        if constexpr (Op::IsScalarOp) {
            TensorType scalar = GetScalar<TensorType, InputOp>();
            opScalars.template Set<pos>(scalar);
            return;
        }
        if constexpr (std::is_same<typename InputOp::DType, uint1_t>::value) {
            static_assert(std::is_same<TensorType, uint8_t>::value,
                          "CopyIn data type is inconsistent with in holder data type.");
            offset = offset / BYTE_LENGTH;
            tileLength = tileLength / BYTE_LENGTH;
        } else {
            static_assert(std::is_same<typename InputOp::DType, TensorType>::value,
                          "CopyIn data type is inconsistent with in holder data type.");
        }

        // Prepare input args
        uint8_t bufId = GetBufId<pos>(pingPong);
        LocalTensor<TensorType> inTensor(AscendC::TPosition::VECIN, bufId * blockLen, blockLen / sizeof(TensorType));
        GlobalTensor<TensorType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ TensorType*>(inGm[InputOp::Pos].GetPhyAddr(offset * sizeof(TensorType))));
        // Set getBuf
        GetTensor<TPosition::VECIN>(bufId);
        // Run copyIn
        Vec::CopyIn<TensorType>(inTensor, globalTensor, tileLength);
        // Set rlsBuf
        ReleaseTensor<TPosition::VECIN>(bufId);
    }

    /**
     * 输出Op
     * @tparam Op CopyOut的Bind节点
     * @tparam pos CopyOut在FunList内的索引
     * @param offset 当前处理的数据在GM上的偏移
     * @param tileLength 当前处理的数据块大小
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
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
            offset = offset / BYTE_LENGTH;
            tileLength = tileLength / BYTE_LENGTH;
        } else {
            static_assert(std::is_same<typename output::DType, inputType>::value,
                          "CopyOut data type is inconsistent with Op data type.");
        }

        // Prepare input args
        uint8_t bufId = GetBufId<GetFunOutputPos<input>()>(pingPong);
        LocalTensor<inputType> localTensor(AscendC::TPosition::VECOUT, bufId * blockLen, blockLen / sizeof(inputType));
        static_assert(output::Pos < outputNums, "output Pos is not less than output number.");
        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(outGm[output::Pos].GetPhyAddr(offset * sizeof(inputType))));
        // Set getBuf
        GetTensor<TPosition::VECOUT>(bufId);
        // Run func
        Vec::CopyOut<inputType>(globalTensor, localTensor, tileLength);
        // Set rlsBuf
        ReleaseTensor<TPosition::VECOUT>(bufId);
    }

    /**
     * 遍历DAG，计算Op的position
     * @tparam Op 待计算的Op, 需要是Bind类型
     * @tparam start 查询起始位置的Position，默认从FunList的第0个开始匹配
     * @return Op在FunList中的索引坐标
     */
    template <class Op, int start = 0>
    __aicore__ constexpr static inline int GetFunOutputPos()
    {
        if constexpr (std::is_same<typename ElemDag::FunList::template At<start>, Op>::value) {
            return start;
        } else if constexpr (start + 1 < ElemDag::FunList::Size) {
            return GetFunOutputPos<Op, start + 1>();
        }
        static_assert(start + 1 < ElemDag::FunList::Size, "The required output in FunList is not found.");
        return -1;
    }

    template <int pos>
    __aicore__ constexpr static inline uint8_t GetBufId(int32_t pingPong)
    {
        if constexpr (bufferIds[0][pos] == bufferIds[1][pos]) {
            return bufferIds[0][pos];
        } else {
            return pingPong == 0 ? bufferIds[0][pos] : bufferIds[1][pos];
        }
    }

    /**
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
            ScalarType scalar = scalars.template Get<scalarValue::Pos>();
            return scalar;
        } else if constexpr (Placeholder::IsInHolder<scalarValue>::Value) {
            GlobalTensor<ScalarType> globalTensor;
            globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalarType*>(inGm[scalarValue::Pos].GetPhyAddr(0)));
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
                TensorType scalar = opScalars.template Get<GetFunOutputPos<InputOp>()>();
                return scalar;
            } else {
                // 判断当前输入是否是同一个操作的多引用输出，避免重复插入同步
                constexpr static bool isDuplicate = Op::InArgs::template IsExist<InputOp, argPos + 1>();
                uint8_t bufId = GetBufId<GetFunOutputPos<InputOp>()>(pingPong);
                if constexpr (!isDuplicate) {
                    GetTensor<TPosition::VECCALC>(bufId);
                }
                LocalTensor<TensorType> inputTensor(AscendC::TPosition::VECCALC, bufId * blockLen,
                                                    blockLen / sizeof(TensorType));
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
                uint8_t bufId = GetBufId<GetFunOutputPos<InputOp>()>(pingPong);
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

    /**
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

    /**
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
        uint8_t bufId = GetBufId<pos>(pingPong);
        LocalTensor<OutputType> outTensor(AscendC::TPosition::VECCALC, bufId * blockLen, blockLen / sizeof(OutputType));
        // Set getBuf
        GetTensor<TPosition::VECCALC>(bufId);
        // Run Op
        RunOp<OutputType, Op, pos>(outTensor, tileLength, pingPong);
        // Set rlsBuf
        ReleaseTensor<TPosition::VECCALC>(bufId);
    }

    /**
     * 执行Scalar节点
     * @tparam Scalar Op的节点
     * @tparam pos 当前计算节点在FunList中的索引
     * @param tileLength 当前计算的数据块大小
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return
     */
    template <class Op, int pos = 0>
    __aicore__ inline constexpr void RunScalarOp(uint64_t tileLength, int32_t pingPong)
    {
        // Prepare output args
        using OutputType = typename Op::template FunRetArgType<0>;
        OutputType outScalar;
        using Func = typename Op::Fun;
        auto inputArgs = PrepareArgs<Op>(pingPong);
        Call<Func, OutputType>(outScalar, inputArgs, tileLength);
        opScalars.template Set<pos>(outScalar);
    }

    // 遍历执行图
    template <int pos = 0>
    __aicore__ inline void Run(uint64_t offset, uint64_t tileLength, int32_t pingPong)
    {
        // Run current func
        using Op = typename ElemDag::FunList::template At<pos>;
        using Func = typename Op::Fun;
        RUN_LOG("RUN.Func[%s]: ArgsSize: %ld, PingPong:%ld, GmOffset:%ld, TileLength:%ld\n", PRINT_TYPE(Func),
                Op::Args::Size, pingPong, offset, tileLength);
        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
            CopyIn<Op, pos>(offset, tileLength, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
            CopyOut<Op, pos>(offset, tileLength, pingPong);
        } else if constexpr (Op::IsScalarOp) {
            RunScalarOp<Op, pos>(tileLength, pingPong);
        } else {
            RunNormalOp<Op, pos>(tileLength, pingPong);
        }

        // Run next func
        if constexpr (pos + 1 < ElemDag::FunList::Size) {
            Run<pos + 1>(offset, tileLength, pingPong);
        }
    }

private:
    constexpr static uint64_t BYTE_LENGTH = 8;
    constexpr static uint64_t CACHE_LINE_BYTE_LENGTH = 512;
    constexpr static int inputNums = ElemDag::InputSize;
    constexpr static int outputNums = ElemDag::OutputSize;
    constexpr static auto bufferIds = ElemDag::template GetBufferIds<true, false>();
    GlobalTensor<uint8_t> inGm[inputNums];
    GlobalTensor<uint8_t> outGm[outputNums];
    TPipe* pipePtr;
    int blockLen = 0;

    int64_t dim0_ = 0;
    int64_t ubFormer_ = 0;
    int64_t blockFormer_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockTail_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubLoopOfFormerBlock_ = 0;
    int64_t ubLoopOfTailBlock_ = 0;
    int64_t ubTailOfFormerBlock_ = 0;
    int64_t ubTailOfTailBlock_ = 0;

    __tiling_data_ptr__ EleBaseTilingData16B* tilingData;
    typename ElemDag::VarType scalars;
    typename ElemDag::ScalarOpType opScalars;
};

} // namespace Base
} // namespace Ops

#endif // ELEWISE_SCH_16B_H_