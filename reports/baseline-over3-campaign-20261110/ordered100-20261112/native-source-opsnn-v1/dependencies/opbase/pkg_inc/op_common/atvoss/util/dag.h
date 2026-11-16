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
 * \file dag.h
 * \brief
 */
#ifndef UTIL_DAG_H_
#define UTIL_DAG_H_
#include "aux1.h"
#include "buffer.h"
#include "node.h"

namespace Ops {
namespace Base {

#ifdef __ATP_UT__
// force to Level_X as specified for Level0
constexpr static int MAX_BUFFER_NUMBER = 0;
#else
constexpr static int MAX_BUFFER_NUMBER = 10;
#endif

/*
 * 若非Bind类型，则原封返回
 */
template <class T, typename = void>
struct GetRealArgs {
    using Type = T;
};

/*
 * 检测Bind入参中Cast节点是否时相同Dtype中间Cast（NoOp）。
 * 若是，则递归获取Cast节点的第一个输入Bind作为当前Bind的入参。
 */
template <class T>
struct GetRealArgs<T, __aux::Void_t<typename T::RealArgs>> {
    using Type = __aux::Condition<(Vec::IsCastNoOp<typename T::Fun>::Value),
                                  typename GetRealArgs<typename T::RealArgs::template At<0>>::Type,
                                  typename T::RealBindType>;
};

/**
 * @tparam MemLevel
 * @tparam ReuseIn
 */
template <MemLevel MemLvl = MemLevel::LEVEL_0, bool ReuseIn = false>
struct MemOptCfg {
    static constexpr MemLevel memoryLevel = MemLvl;
    static constexpr bool reuseIn = ReuseIn;
};

/**
 * @tparam Func 需要执行的向量操作函数
 * @tparam Ts Func的入参
 */
template <typename Func, typename... Ts>
struct Bind {
public:
    using Fun = Func;
    using Args = Elems<Ts...>;
    using BindType = Bind<Fun, Ts...>;
    constexpr static bool IsBindFun = true;

    using RealArgs = typename Args::template ForEach<GetRealArgs>;

private:
    template <typename... Rs>
    struct CreateRealBind;

    template <typename... Rs>
    struct CreateRealBind<Elems<Rs...>> {
        using Type = __aux::Condition<(Vec::IsCastNoOp<Func>::Value), typename RealArgs::template At<0>,
                                      Bind<Func, Rs...>>;
    };

public:
    // 消除入参中CastNoOp之后的BindType
    using RealBindType = typename CreateRealBind<RealArgs>::Type;

    // 第 @offset 个出参Dtype
    template <int offset>
    using FunRetArgType = typename Fun::template FunRetArgType<offset>;
    // 第 @offset 个入参Dtype
    template <int offset>
    using FunInArgType = typename Fun::template FunInArgType<offset>;

public:
    // 当前Bind的输出Dtype
    using OutDataType = FunRetArgType<0>;
    using DType = OutDataType;

    // 入参中输出Placeholder的列表（输出GM)
    using OutHolders = typename Args::template Filter<__aux::TypeIsOutHolder>;
    // 入参列表（Args中过滤掉Placeholder::Out)
    using InArgs = typename Args::template Filter<__aux::TypeIsInput>;
    // 入参中输入Placeholder的列表（输入GM）
    using InHolders = typename Args::template Filter<__aux::TypeIsInHolder>;
    // 入参中函数列表（其他Bind类型）
    using InFuns = typename Args::template Filter<__aux::TypeIsInFun>;
    // 入参中Scalar变量列表
    using Vars = typename Args::template Filter<__aux::TypeIsVar>;
    // 入参中Const输入列表
    using ConstValues = typename Args::template Filter<__aux::TypeIsConst>;
    // 入参中输入是TensorScalar的列表
    using InScalarHolders = typename InHolders::template Filter<__aux::TypeIsInScalarHolder>;
    // 入参中函数是Scalar操作的列表
    using InScalarFuns = typename InFuns::template Filter<__aux::TypeIsScalarBind>;
    // 入参中ReduceOp操作列表
    using InReduceOpFuns = typename InFuns::template Filter<__aux::TypeIsReduceOpBind>;
    // 入参中非ScalarOp函数列表
    using InNonScalarFuns = typename InFuns::template Remove<InScalarFuns>::template Remove<InReduceOpFuns>;

    static_assert(InArgs::Size == InHolders::Size + InFuns::Size + Vars::Size + ConstValues::Size, "why?");

    // 标识当前Bind是否是Scalar操作（不需要使用任何UB空间）
    constexpr static bool IsScalarOp = InScalarHolders::Size == InHolders::Size && InScalarFuns::Size == InFuns::Size &&
                                       !Vec::IsDuplicateOp<Fun>::Value;

    // 入参个数
    constexpr static uint32_t InputSize = InArgs::Size;
    // 当前Bind的依赖列表：输入依赖 + 自身
    using DependFuns = typename __aux::GetDependFunsAux<InFuns>::Type::template Append<RealBindType>;
    // 溯源当前Bind的输入PlaceHolder
    using SourceInHolders = typename __aux::GetSourceInHoldersAux<InFuns>::Type::template Union<InHolders>;
};

/*
 * 获取Bind上的RealBindType
 */
template <class T, typename = void>
struct GetRealBindType {
    using Type = T;
};

template <class T>
struct GetRealBindType<T, __aux::Void_t<typename T::RealBindType>> {
    using Type = typename T::RealBindType;
};

template <class Target, class T>
struct CheckVecReduce {
    constexpr static bool Value = Vec::IsReduceOp<typename T::Fun>::Value;
};

/*
 * DAG 单向无环图处理
 * 模板参数：
 *   1. OutList_:      单向无环图的输出列表
 *   2. ComputeOrder_: 单向无环图函数执行列表，默认空。用户可指定执行序
 *   3. MemOptCfg_:    内存复用策略配置
 */
template <typename OutList_, typename ComputeOrder_ = void, typename MemOptCfg_ = MemOptCfg<>>
struct DAGSch {
public:
    using OutList = typename OutList_::template ForEach<GetRealBindType>;
    using MemOpt = MemOptCfg_;
    static_assert(MemOpt::memoryLevel == MemLevel::LEVEL_0 || MemOpt::memoryLevel == MemLevel::LEVEL_1 ||
                      MemOpt::memoryLevel == MemLevel::LEVEL_2,
                  "Buffer level should be in [0, 1, 2].");

    constexpr static bool HasComputeOrder = !__aux::IsSameType<ComputeOrder_, void>::Value;

private:
    // 通过输出列表，反向推导调用顺序列表
    using FunsAux = typename OutList::template Export<__aux::FunListAux>::Type;
    using FunListOriginal = __aux::Condition<__aux::IsSameType<ComputeOrder_, void>::Value, FunsAux, ComputeOrder_>;

public:
    // Filter CastNoOp
    using FunList = typename FunListOriginal::template ForEach<GetRealBindType>::Unique;

    constexpr static int32_t ReduceOpPos = __aux::GetReducePosition<FunList>();

#ifdef __ATP_UT__
public:
#else
private:
#endif
    using FullNodeInfo = DagNodeInfo<FunList, OutList>;

public:
    // TD: fix me.
    using VecPreReduceNodes = typename __aux::FilterDagAux<FunList, ReduceOpPos, 0, __aux::CheckPre>::Type;
    using VecPostReduceNodes = typename __aux::FilterDagAux<FunList, ReduceOpPos, 0, __aux::CheckPost>::Type;
    using PreReduceNodeInfo = __aux::Condition<
        (ReduceOpPos > 0),
        DagNodeInfo<VecPreReduceNodes, Elems<typename FunList::template At<ReduceOpPos>>,
                    /*supportBrc=*/false>,
        FullNodeInfo>;
    using PostReduceNodeInfo = __aux::Condition<
        (ReduceOpPos > 0), DagNodeInfo<VecPostReduceNodes, OutList, /*supportBrc=*/false>, FullNodeInfo>;

public:
    // 统计 输入/输出 PlaceHolder 列表. 类型是Elems<PlaceHolder<In0>, PlaceHolder<In1>, ...>
    using InHolders = typename FullNodeInfo::InHolders;
    using OutHolders = typename FullNodeInfo::OutHolders;

    // 统计 CopyInBrc 及 VecBrc 列表
    using CopyBrcNodes = typename FullNodeInfo::CopyBrcNodes;
    using VecBrcNodes = typename FullNodeInfo::VecBrcNodes;

    // 统计 输入是 TensorScalar的列表
    using InScalarHolders = typename FullNodeInfo::InScalarHolders;
    // FunList中ScalarOp列表
    using ScalarOpNodes = typename FullNodeInfo::ScalarOpNodes;

    // 统计 输入/输出 GM 数量； CopyInBrc 及 VecBrc 数量
    constexpr static uint32_t InputSize = FullNodeInfo::InputSize;
    constexpr static uint32_t OutputSize = FullNodeInfo::OutputSize;
    constexpr static uint32_t CopyBrcSize = FullNodeInfo::CopyBrcSize;
    constexpr static uint32_t VecBrcSize = FullNodeInfo::VecBrcSize;
    constexpr static uint32_t TensorScalarSize = FullNodeInfo::TensorScalarSize;
    // 刨去 TensorScalar 后的 输入GM 数量
    constexpr static uint32_t InputSizeWoScalar = FullNodeInfo::InputSizeWoScalar;

    // Max `Pos` of InHolders. NOTE: -1 if InHolders is empty.
    // REMEMBER: `InputMaxPos + 1` may NOT equal to InputSize if PlaceHolder::In is NOT contiguous.
    constexpr static int32_t InputMaxPos = FullNodeInfo::InputMaxPos;

    // Scalar
    using Vars = typename FullNodeInfo::Vars;
    using VarType = typename FullNodeInfo::VarType;
    constexpr static uint32_t VarSize = FullNodeInfo::VarSize;
    // ScalarOpType
    using ScalarOpType = typename FullNodeInfo::ScalarOpType;

    // 检查某个输入节点是否可以释放（删除）
    template <int posInFuns, typename InArg>
    constexpr static bool ChkInputCanFree = __aux::InputIsAbleToFreeAux<FunList, posInFuns, InArg>();

    // 检查某个节点是否直连搬出节点.
    template <int posInFuns, typename InArg>
    constexpr static bool IsConnectOutput = __aux::IsConnectOutput<FunList, posInFuns, InArg>();

    // 某个CopyIn节点@InFun，若被后续某个VecBrc节点依赖，则返回VecBrc在VecBrcNodes中的索引位置，否则返回-1
    template <typename InFun>
    constexpr static int VecBrcIdxDepend = FullNodeInfo::template VecBrcIdxDepend<InFun>;

    constexpr static auto MaxAliveNodeInfo = FullNodeInfo::MaxAliveNodeInfo;
    // 最大存活节点数量
    constexpr static uint32_t MaxAliveNode = MaxAliveNodeInfo.aliveNode;
    // 刨去 输入/输出 占用的Buffer，中间计算占用的最大临时节点数量
    constexpr static uint32_t TempCalcNode = MaxAliveNodeInfo.tempCalcNode;
    // 计算途中最大/最小字节数
    constexpr static uint32_t MaxDtypeBytes = MaxAliveNodeInfo.maxDtypeBytes;
    constexpr static uint32_t MinDtypeBytes = MaxAliveNodeInfo.minDtypeBytes;
    // NDDMA场景下最大存活节点数量 及 中间计算占用的最大临时节点数量（刨去了CopyInBrc上的TempNode）
    constexpr static uint32_t MaxAliveNodeForNddma = MaxAliveNodeInfo.aliveNodeNoCopyBrcTmpBuf;
    constexpr static uint32_t TempCalcNodeForNddma = MaxAliveNodeInfo.tempCalcNodeNoCopyBrcTmpBuf;

private:
    constexpr static auto MaxAliveNodeInfoForCacheBrc = FullNodeInfo::MaxAliveNodeInfoForCacheBrc;
    constexpr static auto MaxAliveNodeInfoForPreReduce = PreReduceNodeInfo::MaxAliveNodeInfo;
    constexpr static auto MaxAliveNodeInfoForPostReduce = PostReduceNodeInfo::MaxAliveNodeInfo;

public:
    // CacheBrc场景下 存活节点/中间计算占用的临时节点数量统计 (without Cache Nodes)
    constexpr static uint32_t MaxAliveNodeForCacheBrc = MaxAliveNodeInfoForCacheBrc.aliveNode;
    constexpr static uint32_t TempCalcNodeForCacheBrc = MaxAliveNodeInfoForCacheBrc.tempCalcNode;
    constexpr static uint32_t MaxAliveNodeForNddmaCacheBrc = MaxAliveNodeInfoForCacheBrc.aliveNodeNoCopyBrcTmpBuf;
    constexpr static uint32_t TempCalcNodeForNddmaCacheBrc = MaxAliveNodeInfoForCacheBrc.tempCalcNodeNoCopyBrcTmpBuf;
    constexpr static uint32_t PreReduceAliveNode = MaxAliveNodeInfoForPreReduce.aliveNode;
    constexpr static uint32_t PreReduceTempCalcNode = MaxAliveNodeInfoForPreReduce.tempCalcNode;
    constexpr static uint32_t PostReduceAliveNode = MaxAliveNodeInfoForPostReduce.aliveNode;
    constexpr static uint32_t PostReduceTempCalcNode = MaxAliveNodeInfoForPostReduce.tempCalcNode;

#ifdef __ATP_UT__
public:
#else
private:
#endif
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static MemLevel ChooseBufferLevelImpl()
    {
        if constexpr (MemOpt::memoryLevel == MemLevel::LEVEL_0) {
            if constexpr ((NodeInfo::template GetBufferNumLevel2<use_nddma, cache_brc>()) <= MAX_BUFFER_NUMBER) {
                return MemLevel::LEVEL_2;
            } else if constexpr ((NodeInfo::template GetBufferNumLevel1<use_nddma, cache_brc>()) <= MAX_BUFFER_NUMBER) {
                return MemLevel::LEVEL_1;
            } else {
                return MemLevel::LEVEL_0;
            }
        } else {
            return MemOpt::memoryLevel;
        }
    }

public:
    constexpr static MemLevel BufLevel = ChooseBufferLevelImpl<FullNodeInfo, true, false>();

    // Get All Mte2 Num including Persist ones in @NodeInfo.
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetMte2NumImpl()
    {
        if constexpr (ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>() == MemLevel::LEVEL_0) {
            return (NodeInfo::template GetGMCountBeforeFirstCalcNode<use_nddma, cache_brc>()) +
                   (NodeInfo::template GetPersistMte2Num<use_nddma, cache_brc>());
        } else {
            return NodeInfo::InputSizeWoScalar;
        }
    }

    // Get All Mte2 Num including Persist ones in `FullNodeInfo`
    template <bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetMte2Num()
    {
        return GetMte2NumImpl<FullNodeInfo, use_nddma, cache_brc>();
    }

    // Get All Mte3 Num including Persist ones in @NodeInfo.
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetMte3NumImpl()
    {
        constexpr uint32_t persistMte3Num = NodeInfo::template GetPersistMte3Num<use_nddma, cache_brc>();
        if constexpr (ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>() == MemLevel::LEVEL_0) {
            return NodeInfo::template GetFirstCopyOutNodeGMCount<use_nddma, cache_brc>() + persistMte3Num;
        } else {
            return NodeInfo::template GetLvl12Mte3Count<use_nddma, cache_brc>() + persistMte3Num;
        }
    }

    // Get All Mte3 Num including Persist ones in `FullNodeInfo`
    template <bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetMte3Num()
    {
        return GetMte3NumImpl<FullNodeInfo, use_nddma, cache_brc>();
    }

    // Get All Temp Buff Num including Persist ones in @NodeInfo.
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetTempBufNumImpl()
    {
        constexpr MemLevel bufferLvl = ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>();
        constexpr uint32_t persistTempCalcBufNum = NodeInfo::template GetPersistTempCalcBufNum<use_nddma, cache_brc>();
        if constexpr (bufferLvl == MemLevel::LEVEL_0) {
            return (NodeInfo::template GetLvl0TmpSize<use_nddma, cache_brc>()) + persistTempCalcBufNum;
        } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
            return (NodeInfo::template GetLvl1TmpSize<use_nddma, cache_brc>()) + persistTempCalcBufNum;
        } else {
            return (NodeInfo::template GetTempCalcNodeSize<use_nddma, cache_brc>()) + persistTempCalcBufNum;
        }
    }

    // Get All Temp Buff including Persist ones in `FullNodeInfo`
    template <bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetTempBufNum()
    {
        return GetTempBufNumImpl<FullNodeInfo, use_nddma, cache_brc>();
    }

public:
    // Get All Buff Num including Persist ones in @NodeInfo.
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetBufferNumImpl()
    {
        constexpr MemLevel bufferLvl = ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>();
        if constexpr (bufferLvl == MemLevel::LEVEL_0) {
            return NodeInfo::template GetBufferNumLevel0<use_nddma, cache_brc>();
        } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
            return NodeInfo::template GetBufferNumLevel1<use_nddma, cache_brc>();
        } else { // bufferLvl == 2
            return NodeInfo::template GetBufferNumLevel2<use_nddma, cache_brc>();
        }
    }

    // Get All Buff Num including Persist ones in `FullNodeInfo`.
    template <bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static uint32_t GetBufferNum()
    {
        return GetBufferNumImpl<FullNodeInfo, use_nddma, cache_brc>();
    }

public:
    constexpr static uint32_t BufferNum = GetBufferNumImpl<FullNodeInfo, true, false>();
    constexpr static uint32_t Mte2Num = GetMte2NumImpl<FullNodeInfo, true, false>();
    constexpr static uint32_t Mte3Num = GetMte3NumImpl<FullNodeInfo, true, false>();

public:
    template <typename NodeInfo, bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static const int32_t* const* GetBufferIdsImpl()
    {
        // |mte2|mte3|tmp|mte2|mte3|
        constexpr MemLevel bufferLvl = ChooseBufferLevelImpl<NodeInfo, use_nddma, cache_brc>();
        // Total Mte2 Count including Persist Mte2 in CopyInBrc
        constexpr uint32_t mte2Count = GetMte2NumImpl<NodeInfo, use_nddma, cache_brc>();
        // Total Mte3 Count including Persist Mte3 if CopyInBrc or VecBrc connects to CopyOut
        constexpr uint32_t mte3Count = GetMte3NumImpl<NodeInfo, use_nddma, cache_brc>();
        // Total Temp Buf Count including Cache Temp buf in CopyInBrc & VecBrc
        constexpr uint32_t tempBufCount = GetTempBufNumImpl<NodeInfo, use_nddma, cache_brc>();
        static_assert(((mte2Count + mte3Count) * BUF_PING_PONG + tempBufCount) <= BUF_MAX_COUNT,
                      "Buffer count exceeded 32. Please try to switch MemLevel to LEVEL_1 or LEVEL_0.");
        constexpr uint32_t PongOffset = mte2Count + mte3Count + tempBufCount;
        // Persist Mte2 Count
        constexpr uint32_t persistMte2Count = NodeInfo::template GetPersistMte2Num<use_nddma, cache_brc>();
        // Persist Mte3 Count
        constexpr uint32_t persistMte3Count = NodeInfo::template GetPersistMte3Num<use_nddma, cache_brc>();
        // Persist Temp Buf Count
        constexpr uint32_t persistTempBufCount = NodeInfo::template GetPersistTempCalcBufNum<use_nddma, cache_brc>();
        // Generate Buffer ID List.
        using PersistMte2Es = typename GenerateBufferWrappers<persistMte2Count, BUF_TYPE_MTE2,
                                                              /*Offset=*/0>::Type;
        using Mte2Es = typename GenerateBufferWrappers<mte2Count - persistMte2Count, BUF_TYPE_MTE2,
                                                       /*Offset=*/persistMte2Count>::Type;
        using PersistMte3Es = typename GenerateBufferWrappers<persistMte3Count, BUF_TYPE_MTE3,
                                                              /*Offset=*/mte2Count>::Type;
        using Mte3Es = typename GenerateBufferWrappers<mte3Count - persistMte3Count, BUF_TYPE_MTE3,
                                                       /*Offset=*/mte2Count + persistMte3Count>::Type;
        using PersistTmpEs = typename GenerateBufferWrappers<persistTempBufCount, BUF_TYPE_TEMP,
                                                             /*Offset=*/mte2Count + mte3Count>::Type;
        using TmpEs = typename GenerateBufferWrappers<tempBufCount - persistTempBufCount, BUF_TYPE_TEMP,
                                                      /*Offset=*/mte2Count + mte3Count + persistTempBufCount>::Type;
        using PongMte3Es = __aux::Condition<
            bufferLvl == MemLevel::LEVEL_0,
            typename GenerateBufferWrappers<
                mte3Count - persistMte3Count, BUF_TYPE_MTE3,
                /*Offset=*/mte2Count * BUF_PING_PONG + mte3Count + tempBufCount + persistMte3Count, BUF_PONG>::Type,
            Elems<>>;
        return GenerateBufferIdOrder<
            typename NodeInfo::SavedFunList,
            Elems<PersistMte2Es, Mte2Es, PersistMte3Es, Mte3Es, PersistTmpEs, TmpEs, PongMte3Es>, PongOffset, bufferLvl,
            use_nddma, cache_brc>();
    }

    template <bool use_nddma = true, bool cache_brc = false>
    __aicore__ constexpr static const int32_t* const* GetBufferIds()
    {
        return GetBufferIdsImpl<FullNodeInfo, use_nddma, cache_brc>();
    }

    template <bool preReduce = true>
    __aicore__ constexpr static const int32_t* const* GetReduceBufferIds()
    {
        static_assert(ReduceOpPos > 0, "ReduceOp should exist in DAG when try to call GetReduceBufferIds.");
        if constexpr (preReduce) {
            if constexpr (ReduceOpPos == 1 && Vec::IsCopyInOp<typename VecPreReduceNodes::template At<0>::Fun>::Value) {
                return PreReduceOnlyCopyInBufferId::Value;
            } else {
                return GetBufferIdsImpl<PreReduceNodeInfo, true, false>();
            }
        } else {
            return GetBufferIdsImpl<PostReduceNodeInfo, true, false>();
            ;
        }
    }
};

} // namespace Base
} // namespace Ops
#endif // UTIL_DAG_H_
