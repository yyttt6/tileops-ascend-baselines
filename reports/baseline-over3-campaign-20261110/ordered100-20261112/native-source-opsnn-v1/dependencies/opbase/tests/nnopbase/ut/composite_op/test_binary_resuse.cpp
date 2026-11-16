/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "kernel_launcher.h"
#include "op_run_context.h"
#include "op_ctx_def.h"
#include "opdev/make_op_executor.h"
#include "utils/file_faker.h"
#include "thread_local_context.h"
#include "kernel_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus NnopbaseInit();
#ifdef __cplusplus
}
#endif

using namespace op;
using namespace std;
OP_TYPE_REGISTER(TestStaticAdd)
OP_TYPE_REGISTER(Conv2D)
OP_TYPE_REGISTER(Gelu)

namespace op {
extern uint32_t GenOpTypeId(const char* op_name, const OP_RESOURCES& op_resources);
extern uint32_t GenOpTypeId(const char* op_name, const OP_SOC_RESOURCES& op_resources);
} // namespace op

static int TestGenSimplifiedKey(gert::TilingContext* ctx, char* simplifiedKey)
{
    std::strcat(simplifiedKey, "p=[-1,null]/float16");
    return 0;
}

static void FakeTilingFunc() {}

static OP_RESOURCES GenerateFakeOpResource()
{
    std::vector<void*> tilingFuncs;
    tilingFuncs.emplace_back(reinterpret_cast<void*>(&FakeTilingFunc));

    static std::string opDescJson = " ";
    static std::string opBinJson = " ";
    static std::string opBinData = "123";
    std::vector<std::tuple<const uint8_t*, const uint8_t*>> binInfo;
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opDescJson.c_str()),
                                         reinterpret_cast<const uint8_t*>(opDescJson.c_str() + opDescJson.size())));
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opBinJson.c_str()),
                                         reinterpret_cast<const uint8_t*>(opBinJson.c_str() + opBinJson.size())));
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opBinData.c_str()),
                                         reinterpret_cast<const uint8_t*>(opBinData.c_str() + opBinData.size())));

    static std::string rk = "456";
    std::vector<std::tuple<const uint8_t*, const uint8_t*>> runtimeKb;
    runtimeKb.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(rk.c_str()),
                                           reinterpret_cast<const uint8_t*>(rk.c_str() + rk.size())));

    return {{"fake2", std::make_tuple(tilingFuncs, binInfo, runtimeKb)},
            {"x", std::make_tuple(tilingFuncs, binInfo, runtimeKb)}};
}

static OP_SOC_RESOURCES GenerateFakeOpSocResource(std::string& socVersion)
{
    std::vector<void*> tilingFuncs;
    tilingFuncs.emplace_back(reinterpret_cast<void*>(&FakeTilingFunc));

    static std::string opDescJson = " ";
    static std::string opBinJson = " ";
    static std::string opBinData = "123";
    std::vector<std::tuple<const uint8_t*, const uint8_t*>> binInfo;
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opDescJson.c_str()),
                                         reinterpret_cast<const uint8_t*>(opDescJson.c_str() + opDescJson.size())));
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opBinJson.c_str()),
                                         reinterpret_cast<const uint8_t*>(opBinJson.c_str() + opBinJson.size())));
    binInfo.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(opBinData.c_str()),
                                         reinterpret_cast<const uint8_t*>(opBinData.c_str() + opBinData.size())));

    std::map<ge::AscendString, std::vector<std::tuple<const uint8_t*, const uint8_t*>>> binMap;
    binMap[socVersion.c_str()] = binInfo;

    static std::string rk = "456";
    std::vector<std::tuple<const uint8_t*, const uint8_t*>> runtimeKb;
    runtimeKb.emplace_back(std::make_tuple(reinterpret_cast<const uint8_t*>(rk.c_str()),
                                           reinterpret_cast<const uint8_t*>(rk.c_str() + rk.size())));

    return {{"fake3", std::make_tuple(tilingFuncs, binMap, runtimeKb)},
            {"x", std::make_tuple(tilingFuncs, binMap, runtimeKb)}};
}

class TestBinaryReuse : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::GenOpTypeId("AddN");
        op::GenOpTypeId("Conv2D");
        op::GenOpTypeId("Conv2DBackpropInput");
        op::GenOpTypeId("Conv2DBackpropFilter");
        op::GenOpTypeId("LayerNormXBackpropV2");
        op::GenOpTypeId("ConcatD");
        op::GenOpTypeId("BNTrainingUpdateGrad");
        op::GenOpTypeId("BNTrainingUpdate");
        op::GenOpTypeId("BNTrainingReduce");
        op::GenOpTypeId("BN3DTrainingReduce");
        op::GenOpTypeId("TransData");
        op::GenOpTypeId("MatMul");
        op::GenOpTypeId("Log");
        op::GenOpTypeId("ViewCopy");
        op::GenOpTypeId("ReduceAll");
        op::GenOpTypeId("GatherV2");
        op::GenOpTypeId("TopKV2");
        op::GenOpTypeId("NLLLoss");
        op::GenOpTypeId("LpNormUpdate");
        op::GenOpTypeId("Renorm");
        op::GenOpTypeId("MemSet");
        op::GenOpTypeId("NonZero");
        op::GenOpTypeId("LayerNormBetaGammaBackpropV2");
        op::GenOpTypeId("TestStaticAdd");
        op::GenOpTypeId("Simplifiedkeytest");
        OP_RESOURCES opResource = GenerateFakeOpResource();
        op::GenOpTypeId("fake1", opResource);
        std::string soc1 = "ascend910";
        OP_SOC_RESOURCES opSocResource1 = GenerateFakeOpSocResource(soc1);
        op::GenOpTypeId("fake2", opSocResource1);
        std::string soc2 = "ascend910b";
        OP_SOC_RESOURCES opSocResource2 = GenerateFakeOpSocResource(soc2);
        op::GenOpTypeId("fake3", opSocResource2);
        GeluOpTypeId();
        for (size_t i = 1; i < OpTypeDict::GetAllOpTypeSize(); ++i) {
            op::internal::KernelMgr::getInstance().AclOpKernelInit(i);
        }
        aclnnStatus gNnopbaseInit = NnopbaseInit();
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
        std::cout << "TestBinaryReuse SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        NnopbaseUnsetEnvAndClearFolder();
        std::cout << "TestBinaryReuse TearDown" << std::endl;
    }

    int64_t GetShapeSize(const vector<int64_t>& shape)
    {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    vector<int64_t> ToContiguousStrides(const vector<int64_t>& shape)
    {
        vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size() - 2); i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        return strides;
    }

    void CheckResult(op::internal::OpKernelBin* bin, const string& filePrefix)
    {
        std::string prefix = "kernel/ascend910";
        if (filePrefix.empty()) {
            ASSERT_EQ(bin, nullptr);
        } else {
            std::string jsonPath = filePrefix + ".json";
            std::string binPath = filePrefix + ".o";

            ASSERT_NE(bin, nullptr);
            if (bin->binType_ == op::internal::BinType::STATIC_BIN) {
                auto posJson = bin->binPath_.find(filePrefix);
                ASSERT_NE(posJson, string::npos);
            } else {
                auto posJson = bin->jsonPath_.find(prefix);
                auto posBin = bin->jsonPath_.find(prefix);
                ASSERT_NE(posJson, string::npos);
                EXPECT_EQ(bin->jsonPath_.substr(posJson), jsonPath);
                EXPECT_EQ(bin->binPath_.substr(posBin), binPath);
            }
        }
    }

#define LAUNCH_AND_CHECK(opTypeId, CORE_TYPE, profilingInfoId, filePrefix, op_args...)                     \
    do {                                                                                                   \
        op::OpArgContext* ctx = op::MakeOpArgContext(op_args);                                             \
        auto* launcher = new op::AiCoreKernelLauncher{opTypeId, CORE_TYPE, profilingInfoId, nullptr, ctx}; \
        auto bin = launcher->GetBin();                                                                     \
        CheckResult(bin, filePrefix);                                                                      \
        delete launcher;                                                                                   \
    } while (0)

    void CreateRunAndCheck(bool withBias, const string& jsonPath, const string& binPath, bool dtypeCorrect = true,
                           bool formatCorrect = true)
    {
        int64_t multiStride1 = 2;
        int64_t multiStride2 = 2;
        int64_t multiStrideResult = 2;
        aclDataType dtype1 = aclDataType::ACL_FLOAT16;
        aclDataType dtype2 = aclDataType::ACL_FLOAT16;
        aclDataType dtypeResult = aclDataType::ACL_FLOAT16;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeB = {2, 3, 32, 16};
        vector<int64_t> shapeResult = shapeB;
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesB = ToContiguousStrides(shapeB);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesB[0] *= multiStride2;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeB = shapeB;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeB[0] *= multiStride2;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        vector<int32_t> dataB(GetShapeSize(storageShapeB) * 2, 3);
        void* deviceDataA = nullptr;
        void* deviceDataB = nullptr;
        void* deviceDataResult = nullptr;

        aclFormat inputAFormat = aclFormat::ACL_FORMAT_NC1HWC0;
        if (!formatCorrect) {
            inputAFormat = aclFormat::ACL_FORMAT_NCHW;
        }
        aclTensor* a = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0, inputAFormat,
                                       storageShapeA.data(), storageShapeA.size(), deviceDataA);
        aclTensor* b = aclCreateTensor(shapeB.data(), shapeB.size(), dtype2, stridesB.data(), 0,
                                       aclFormat::ACL_FORMAT_FRACTAL_Z, storageShapeB.data(), storageShapeB.size(),
                                       deviceDataB);
        aclTensor* bias = nullptr;
        if (withBias) {
            bias = aclCreateTensor(shapeB.data(), shapeB.size(), dtype2, stridesB.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   storageShapeB.data(), storageShapeB.size(), deviceDataB);
        }

        if (!dtypeCorrect) {
            dtypeResult = aclDataType::ACL_FLOAT;
        }
        aclTensor* result = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtypeResult, stridesResult.data(),
                                            0, aclFormat::ACL_FORMAT_NC1HWC0, storageShapeResult.data(),
                                            storageShapeResult.size(), deviceDataResult);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("Conv2D");

        auto ctx = op::MakeOpArgContext(OP_INPUT(a, b, bias), OP_OUTPUT(result));
        auto* launcher = new op::AiCoreKernelLauncher(opType, AI_CORE, profilingId, nullptr, ctx);
        auto bin = launcher->GetBin();

        if (formatCorrect && dtypeCorrect) {
            ASSERT_NE(bin, nullptr);

            std::string prefix = "kernel/ascend910";
            auto posJson = bin->jsonPath_.find(prefix);
            auto posBin = bin->jsonPath_.find(prefix);
            ASSERT_NE(posJson, string::npos);
            EXPECT_EQ(bin->jsonPath_.substr(posJson), jsonPath);
            EXPECT_EQ(bin->binPath_.substr(posBin), binPath);
        } else {
            ASSERT_EQ(bin, nullptr);
        }
        delete launcher;
        delete a;
        delete b;
        delete result;
        if (withBias) {
            delete bias;
        }
    }

    void CreateDynamicInputsAndCheckWithThreeInputs(const string& opTypeStr, aclDataType dataType = ACL_FLOAT16)
    {
        op::DataType dtype = static_cast<op::DataType>(dataType);
        auto unique_executor = CREATE_EXECUTOR();
        aclTensor* tensors[3] = {nullptr};
        for (auto& tensor : tensors) {
            tensor = unique_executor->AllocTensor(dtype, ge::FORMAT_ND, ge::FORMAT_ND);
        }

        aclTensorList* tensorList = unique_executor->AllocTensorList(reinterpret_cast<const aclTensor**>(&tensors), 3);
        aclTensor* result = unique_executor->AllocTensor(dtype, ge::FORMAT_ND, ge::FORMAT_ND);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(opTypeStr);
        string filePrefix = "kernel/ascend910/add_n/AddN_8460c4b727ab5cc93a9595333a3fd571_high_performance";
        if (opTypeStr == "ConcatD") {
            filePrefix = "kernel/ascend910/concat_d/ConcatD_022c2ffc1fc8b7b7fce31a48ed50ca2c_high_performance";
        }

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorList), OP_OUTPUT(result));
    }

    void CreateTransData(ge::Format inputFormat, ge::DataType inputDt, ge::Format outputFormat, ge::DataType outputDt,
                         const string& filePrefix, bool is_soc_910b = false)
    {
        gert::Shape shapeInput = {2, 1, 16, 16};
        if (inputFormat == ge::FORMAT_FRACTAL_Z) {
            if ((inputDt == ge::DT_FLOAT || inputDt == ge::DT_UINT32 || inputDt == ge::DT_INT32) && is_soc_910b) {
                shapeInput = {2, 1, 16, 8};
            } else if (inputDt == ge::DT_INT8) {
                shapeInput = {2, 1, 32, 32};
            }
        }

        gert::Shape shapeResult = {6, 1, 16, 16};
        if (outputFormat == ge::FORMAT_FRACTAL_Z) {
            if ((outputDt == ge::DT_FLOAT || outputDt == ge::DT_UINT32 || outputDt == ge::DT_INT32) && is_soc_910b) {
                shapeResult = {2, 1, 16, 8};
            } else if (outputDt == ge::DT_INT8 || outputDt == ge::DT_UINT8) {
                shapeResult = {2, 1, 32, 32};
            }
        }
        auto unique_executor = CREATE_EXECUTOR();
        aclTensor* tensorIn = unique_executor->AllocTensor(shapeInput, shapeInput, inputDt, inputFormat,
                                                           ge::FORMAT_NCHW);

        aclTensor* tensorOut = unique_executor->AllocTensor(shapeResult, shapeResult, outputDt, outputFormat,
                                                            ge::FORMAT_NCHW);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("TransData");
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorIn), OP_OUTPUT(tensorOut));
    }

    void CreateMatMul(ge::Format inputFormat, ge::DataType inputDt, ge::Format outputFormat, ge::DataType outputDt,
                      const string& filePrefix, bool has_attr = false, bool transposeX1 = true, bool transposeX2 = true)
    {
        auto unique_executor = CREATE_EXECUTOR();

        gert::Shape inputShape({2, 1, 32, 16});
        aclTensor* tensorIn1 = unique_executor->AllocTensor(inputShape, inputShape, inputDt, inputFormat,
                                                            ge::FORMAT_ND);

        aclTensor* tensorIn2 = unique_executor->AllocTensor(inputShape, inputShape, inputDt, inputFormat,
                                                            ge::FORMAT_ND);

        aclTensor* tensorOut = unique_executor->AllocTensor(inputShape, inputShape, outputDt, outputFormat,
                                                            ge::FORMAT_ND);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("MatMul");
        if (has_attr) {
            LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorIn1, tensorIn2),
                             OP_OUTPUT(tensorOut), OP_ATTR(transposeX1, transposeX2));
        } else {
            LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorIn1, tensorIn2),
                             OP_OUTPUT(tensorOut));
        }
    }

    void CreateAddNAndCheckWith20Inputs(ge::DataType dtype, ge::Format format)
    {
        op::DataType dtypeOp = static_cast<op::DataType>(dtype);
        auto unique_executor = CREATE_EXECUTOR();
        aclTensor* tensors[3] = {nullptr};
        for (auto& tensor : tensors) {
            tensor = unique_executor->AllocTensor(dtypeOp, format, ge::FORMAT_ND);
        }

        aclTensorList* tensorList = unique_executor->AllocTensorList(reinterpret_cast<const aclTensor**>(&tensors), 3);
        aclTensor* result = unique_executor->AllocTensor(dtypeOp, format, ge::FORMAT_ND);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("AddN");
        string filePrefix = "kernel/ascend910/add_n/AddN_8460c4b727ab5cc93a9595333a3fd571_high_performance";
        if (dtype == ge::DT_FLOAT) {
            filePrefix = "kernel/ascend910/add_n/AddN_3c918d330f983e55e90427727bb63ca4_high_performance";
        }

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorList), OP_OUTPUT(result));
    }

    template <typename... ATTRS>
    void CreateLayerNormBetaGammaBackpropV2(const string& filePrefix, ATTRS&&... attrs)
    {
        op::DataType dtypeOp = ge::DT_FLOAT;
        auto unique_executor = CREATE_EXECUTOR();

        auto input0 = unique_executor->AllocTensor(dtypeOp, ge::FORMAT_ND, ge::FORMAT_ND);
        auto input1 = unique_executor->AllocTensor(dtypeOp, ge::FORMAT_ND, ge::FORMAT_ND);
        auto output0 = unique_executor->AllocTensor(dtypeOp, ge::FORMAT_ND, ge::FORMAT_ND);
        auto output1 = unique_executor->AllocTensor(dtypeOp, ge::FORMAT_ND, ge::FORMAT_ND);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("LayerNormBetaGammaBackpropV2");

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1),
                         OP_OUTPUT(output0, output1), OP_ATTR(std::forward<ATTRS>(attrs)...));
    }

    void CreateConcatD(aclDataType dtype, aclFormat format, const string& filePrefix)
    {
        int64_t multiStride1 = 2;
        int64_t multiStrideResult = 2;
        aclDataType dtype1 = dtype;
        aclDataType dtypeResult = dtype;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeResult{6, 1, 32, 16};
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        void* deviceDataA = nullptr;
        void* deviceDataResult = nullptr;

        aclTensor* tensors[20] = {nullptr};
        for (auto& tensor : tensors) {
            tensor = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0, format,
                                     storageShapeA.data(), storageShapeA.size(), deviceDataA);
        }

        aclTensorList* tensorList = aclCreateTensorList(reinterpret_cast<const aclTensor**>(&tensors), 3);
        aclTensor* result = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtypeResult, stridesResult.data(),
                                            0, format, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("ConcatD");
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(tensorList), OP_OUTPUT(result));
        for (auto& tensor : tensors) {
            delete tensor;
        }
        delete tensorList;
        delete result;
    }

    template <typename... ATTRS>
    void CreateSingleInAndout(const string& op_type, op::DataType dtype, op::Format format, const string& filePrefix,
                              ATTRS&&... opAttrs)
    {
        auto unique_executor = CREATE_EXECUTOR();

        auto input = unique_executor->AllocTensor(dtype, format, format);
        auto output = unique_executor->AllocTensor(dtype, format, format);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input), OP_OUTPUT(output),
                         OP_ATTR(std::forward<ATTRS>(opAttrs)...));
    }

    template <typename... ATTRS>
    void CreateSingleInAndoutWithShape(const string& op_type, op::DataType dtypeIn, op::DataType dtypeOut,
                                       op::Format format, const op::Shape& shapeIn, const op::Shape& shapeOut,
                                       const string& filePrefix, ATTRS&&... opAttrs)
    {
        auto unique_executor = CREATE_EXECUTOR();

        auto input = unique_executor->AllocTensor(shapeIn, dtypeIn, format);
        auto output = unique_executor->AllocTensor(shapeOut, dtypeOut, format);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input), OP_OUTPUT(output),
                         OP_ATTR(std::forward<ATTRS>(opAttrs)...));
    }

    template <typename... ATTRS>
    void CreateNoInputOutput(const string& op_type, const string& filePrefix, ATTRS&&... opAttrs)
    {
        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_ATTR(std::forward<ATTRS>(opAttrs)...));
    }

    void CreateTopKV2(const string& op_type, aclDataType dtype0, aclDataType dtype1, aclFormat format,
                      const string& filePrefix)
    {
        int64_t multiStride1 = 2;
        int64_t multiStrideResult = 2;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeResult{6, 1, 32, 16};
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        void* deviceDataResult = nullptr;

        aclTensor* input0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                            format, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* input1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype1, stridesResult.data(), 0,
                                            format, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* result0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                             format, storageShapeResult.data(), storageShapeResult.size(),
                                             deviceDataResult);
        aclTensor* result1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype1, stridesResult.data(), 0,
                                             format, storageShapeResult.data(), storageShapeResult.size(),
                                             deviceDataResult);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1),
                         OP_OUTPUT(result0, result1), OP_ATTR(true, 0, true));
        delete input0;
        delete input1;
        delete result0;
        delete result1;
    }

    template <typename... ATTRS>
    void CreateTwoInAndSingleout(const string& op_type, ge::DataType dtype0, ge::DataType dtype1, ge::Format format,
                                 const string& filePrefix, ATTRS&&... attrs)
    {
        auto unique_executor = CREATE_EXECUTOR();

        auto input0 = unique_executor->AllocTensor(dtype0, format, format);
        auto input1 = unique_executor->AllocTensor(dtype1, format, format);
        auto output0 = unique_executor->AllocTensor(dtype0, format, format);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1), OP_OUTPUT(output0),
                         OP_ATTR(std::forward<ATTRS>(attrs)...));
    }

    void TestStaticBinaryAdd(const string& filePrefix)
    {
        op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "TestStaticAdd";
        op::internal::GetThreadLocalContext().logInfo_.l0Name = "TestStaticAdd";
        auto uniqueExecutor = CREATE_EXECUTOR();
        aclOpExecutor* executor = uniqueExecutor.get();
        op::Shape shape0({512, 512, 3, 3});
        auto input0 = executor->AllocTensor(shape0, ge::DT_FLOAT, ge::FORMAT_NCHW);

        auto input1 = executor->AllocTensor(shape0, ge::DT_FLOAT, ge::FORMAT_NCHW);

        auto output = executor->AllocTensor(shape0, ge::DT_FLOAT, ge::FORMAT_NCHW);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("TestStaticAdd");
        {
            /* Test abnormal case. */
            aclTensor* tensors[3] = {nullptr};
            for (auto& tensor : tensors) {
                tensor = executor->AllocTensor(ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
            }

            aclTensorList* tensorList = executor->AllocTensorList(reinterpret_cast<const aclTensor**>(&tensors), 3);
            LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, "", OP_INPUT(nullptr, tensorList), OP_OUTPUT(output));
        }
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1), OP_OUTPUT(output));

        auto ctx = op::MakeOpArgContext(OP_INPUT(input0, input1), OP_OUTPUT(output));
        aclTensorList* workspace = nullptr;
        auto ret = op::internal::GetWorkspace(
            op::OpTypeDict::ToOpType("TestStaticAdd"), &workspace, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
            *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
        op::DestroyOpArgContext(ctx);

        EXPECT_EQ(ret, ACL_SUCCESS);
        ASSERT_NE(workspace, nullptr);
        ASSERT_EQ(workspace->Size(), 3u);
        EXPECT_EQ((*workspace)[0]->GetOriginalShape(), op::Shape{34});
        EXPECT_EQ((*workspace)[1]->GetOriginalShape(), op::Shape{35});
        EXPECT_EQ((*workspace)[2]->GetOriginalShape(), op::Shape{36});

        auto inputTuple = OP_INPUT(input0, input1);
        auto outputTuple = OP_OUTPUT(output);
        auto attrTuple = OP_ATTR();
        auto ws = OP_WORKSPACE(workspace);
        auto ctx2 = op::MakeOpArgContext(inputTuple, outputTuple, attrTuple, ws);
        int dummyStream = 0;
        void* stream = &dummyStream;
        bool original_flag = op::internal::RtsApiFlag::GetRtsApiFlag().IfNewApi();
        OP_LOGD("Use new api %d.", op::internal::RtsApiFlag::GetRtsApiFlag().IfNewApi());
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(false);
        auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx2);
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(original_flag);
        op::DestroyOpArgContext(ctx2);
        EXPECT_EQ(rc, ACL_SUCCESS);
    }

    void TestStaticBinaryConv2D(const string& filePrefix)
    {
        op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "Conv2D";
        op::internal::GetThreadLocalContext().logInfo_.l0Name = "Conv2D";
        auto uniqueExecutor = CREATE_EXECUTOR();
        aclOpExecutor* executor = uniqueExecutor.get();
        op::Shape shape0({32, 4, 96, 96, 16});
        int32_t nc1hwc0FormatWithC0 = 3;
        int32_t fzFormatWithC0 = 4;
        auto input0 = executor->AllocTensor(shape0, ge::DT_FLOAT16, (ge::Format)(nc1hwc0FormatWithC0));
        op::Shape shape1({36, 4, 16, 16});
        auto input1 = executor->AllocTensor(shape1, ge::DT_FLOAT16, (ge::Format)(fzFormatWithC0));
        auto output0 = executor->AllocTensor(shape0, ge::DT_FLOAT16, (ge::Format)(nc1hwc0FormatWithC0));

        int64_t paddings[4] = {1, 1, 1, 1};
        int64_t strides[4] = {1, 1, 1, 1};
        int64_t dilations[4] = {1, 1, 1, 1};
        auto stridesIntArray = executor->AllocIntArray(strides, 4);
        auto paddingsIntArray = executor->AllocIntArray(paddings, 4);
        auto dilationsIntArray = executor->AllocIntArray(dilations, 4);
        int32_t groups = 1;
        string dataFormat = "NCHW";
        int64_t offset_x = 0;

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("Conv2D");
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1), OP_OUTPUT(output0),
                         OP_ATTR(stridesIntArray, paddingsIntArray, dilationsIntArray, groups, dataFormat, offset_x));

        auto input = OP_INPUT(input0, input1);
        auto output = OP_OUTPUT(output0);
        auto opAttr = OP_ATTR(stridesIntArray, paddingsIntArray, dilationsIntArray, groups, dataFormat, offset_x);
        auto ctx = op::MakeOpArgContext(input, output, opAttr);
        aclTensorList* workspace = nullptr;
        auto ret = op::internal::GetWorkspace(
            op::OpTypeDict::ToOpType("Conv2D"), &workspace, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
            *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
        auto ws = OP_WORKSPACE(workspace);
        ctx->AppendOpWorkspaceArg(workspace);

        EXPECT_EQ(ret, ACL_SUCCESS);
        ASSERT_NE(workspace, nullptr);
        ASSERT_EQ(workspace->Size(), 3u);
        EXPECT_EQ((*workspace)[0]->GetOriginalShape(), op::Shape{333});
        EXPECT_EQ((*workspace)[1]->GetOriginalShape(), op::Shape{444});
        EXPECT_EQ((*workspace)[2]->GetOriginalShape(), op::Shape{555});

        int dummyStream = 0;
        void* stream = &dummyStream;
        bool original_flag = op::internal::RtsApiFlag::GetRtsApiFlag().IfNewApi();
        OP_LOGD("Use new api %d.", op::internal::RtsApiFlag::GetRtsApiFlag().IfNewApi());
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(false);
        auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(original_flag);
        op::DestroyOpArgContext(ctx);
        EXPECT_EQ(rc, ACL_SUCCESS);
    }

    void CreateNLLLoss(const string& op_type, aclDataType dtype0, aclDataType dtype1, aclFormat format,
                       size_t input_num, const string& filePrefix, const std::string& attr)
    {
        int64_t multiStride1 = 2;
        int64_t multiStrideResult = 2;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeResult{6, 1, 32, 16};
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        void* deviceDataResult = nullptr;

        aclTensor* input0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                            format, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* input1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype1, stridesResult.data(), 0,
                                            format, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);

        aclTensor* result0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                             format, storageShapeResult.data(), storageShapeResult.size(),
                                             deviceDataResult);
        aclTensor* result1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                             format, storageShapeResult.data(), storageShapeResult.size(),
                                             deviceDataResult);
        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        if (input_num > 2) {
            aclTensor* input2 = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtype0, stridesResult.data(), 0,
                                                format, storageShapeResult.data(), storageShapeResult.size(),
                                                deviceDataResult);
            LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1, input2),
                             OP_OUTPUT(result0, result1), OP_ATTR(attr.c_str()));
            delete input2;
        } else {
            LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1),
                             OP_OUTPUT(result0, result1), OP_ATTR(attr.c_str()));
        }

        delete input0;
        delete input1;
        delete result0;
        delete result1;
    }

    template <typename... ATTRS>
    void CreateConv2DBackpropInput(const string& filePrefix, ATTRS&&... opAttrs)
    {
        int64_t multiStride1 = 2;
        int64_t multiStrideResult = 2;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeResult{6, 1, 32, 16};
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        void* deviceDataResult = nullptr;

        aclTensor* input0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_INT32, stridesResult.data(), 0,
                                            ACL_FORMAT_NCHW, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* input1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT16, stridesResult.data(),
                                            0, ACL_FORMAT_FRACTAL_Z, storageShapeResult.data(),
                                            storageShapeResult.size(), deviceDataResult);
        aclTensor* input2 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT16, stridesResult.data(),
                                            0, ACL_FORMAT_NC1HWC0, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* result = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT16, stridesResult.data(),
                                            0, ACL_FORMAT_NC1HWC0, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("Conv2DBackpropInput");

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1, input2), OP_OUTPUT(result),
                         OP_ATTR(std::forward<ATTRS>(opAttrs)...));

        delete input0;
        delete input1;
        delete input2;
        delete result;
    }

    void CreateConv2DBackpropFilter(const string& filePrefix)
    {
        int64_t multiStride1 = 2;
        int64_t multiStrideResult = 2;

        vector<int64_t> shapeA = {2, 1, 32, 16};
        vector<int64_t> shapeResult{6, 1, 32, 16};
        auto stridesA = ToContiguousStrides(shapeA);
        auto stridesResult = ToContiguousStrides(shapeResult);
        stridesA[0] *= multiStride1;
        stridesResult[0] *= multiStrideResult;
        auto storageShapeA = shapeA;
        auto storageShapeResult = shapeResult;
        storageShapeA[0] *= multiStride1;
        storageShapeResult[0] *= multiStrideResult;
        vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
        void* deviceDataResult = nullptr;

        aclTensor* input0 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT16, stridesResult.data(),
                                            0, ACL_FORMAT_NC1HWC0, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* input1 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_INT32, stridesResult.data(), 0,
                                            ACL_FORMAT_NCHW, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* input2 = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT16, stridesResult.data(),
                                            0, ACL_FORMAT_NC1HWC0, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);
        aclTensor* result = aclCreateTensor(shapeResult.data(), shapeResult.size(), ACL_FLOAT, stridesResult.data(), 0,
                                            ACL_FORMAT_FRACTAL_Z, storageShapeResult.data(), storageShapeResult.size(),
                                            deviceDataResult);

        const char* nchw = "NCHW";
        std::string nchwStr("NCHW");
        std::string nhwcStr("NHWC");
        int64_t value = 1;
        aclIntArray strides(&value, 1);
        aclIntArray pads(&value, 1);
        aclIntArray dilations(&value, 1);
        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType("Conv2DBackpropFilter");

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, OP_INPUT(input0, input1, input2), OP_OUTPUT(result),
                         OP_ATTR(&strides, &pads, &dilations, value, nchw));

        delete input0;
        delete input1;
        delete input2;
        delete result;
    }
};

TEST_F(TestBinaryReuse, testConv2dNoBiasBinaryReuse)
{
    cout << "test aclopConv2d no bias, binaryReuse ..." << endl;

    const std::string jsonPath = "kernel/ascend910/conv2d/Conv2D_NZ_NZ_FP16_FP16_NO_BIAS_all.json";
    const std::string binPath = "kernel/ascend910/conv2d/Conv2D_NZ_NZ_FP16_FP16_NO_BIAS_all.o";
    CreateRunAndCheck(false, jsonPath, binPath);
}

TEST_F(TestBinaryReuse, testConv2dWithBiasBinaryReuse)
{
    cout << "test aclopConv2d with bias, binaryReuse ..." << endl;

    const std::string jsonPath = "kernel/ascend910/conv2d/Conv2D_NZ_NZ_FP16_FP16_WITH_BIAS_all.json";
    const std::string binPath = "kernel/ascend910/conv2d/Conv2D_NZ_NZ_FP16_FP16_WITH_BIAS_all.o";

    CreateRunAndCheck(true, jsonPath, binPath);
}

TEST_F(TestBinaryReuse, testConv2dWithBiasBinaryReuseFormatNotMatch)
{
    cout << "test aclopConv2d with bias and wrong format, binaryReuse not match..." << endl;

    CreateRunAndCheck(true, "", "", true, false);
}

TEST_F(TestBinaryReuse, testConv2dWithBiasBinaryReuseDtypeNotMatch)
{
    cout << "test aclopConv2d with bias and wrong dtype, binaryReuse not match ..." << endl;
    CreateRunAndCheck(true, "", "", false, true);
}

TEST_F(TestBinaryReuse, testAddnBinaryReuseWithoutAcltensorList)
{
    cout << "test aclop AddN binaryReuse ..." << endl;
    int64_t multiStride1 = 2;
    int64_t multiStride2 = 2;
    int64_t multiStrideResult = 2;
    aclDataType dtype1 = aclDataType::ACL_FLOAT16;
    aclDataType dtype2 = aclDataType::ACL_FLOAT16;
    aclDataType dtypeResult = aclDataType::ACL_FLOAT16;

    vector<int64_t> shapeA = {2, 1, 32, 16};
    vector<int64_t> shapeB = {2, 3, 32, 16};
    vector<int64_t> shapeResult = shapeB;
    auto stridesA = ToContiguousStrides(shapeA);
    auto stridesB = ToContiguousStrides(shapeB);
    auto stridesResult = ToContiguousStrides(shapeResult);
    stridesA[0] *= multiStride1;
    stridesB[0] *= multiStride2;
    stridesResult[0] *= multiStrideResult;
    auto storageShapeA = shapeA;
    auto storageShapeB = shapeB;
    auto storageShapeResult = shapeResult;
    storageShapeA[0] *= multiStride1;
    storageShapeB[0] *= multiStride2;
    storageShapeResult[0] *= multiStrideResult;
    vector<int32_t> dataA(GetShapeSize(storageShapeA) * 2, 2);
    vector<int32_t> dataB(GetShapeSize(storageShapeB) * 2, 3);
    void* deviceDataA = nullptr;
    void* deviceDataB = nullptr;
    void* deviceDataResult = nullptr;

    aclTensor* a = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   storageShapeA.data(), storageShapeA.size(), deviceDataA);
    aclTensor* b = aclCreateTensor(shapeB.data(), shapeB.size(), dtype2, stridesB.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   storageShapeB.data(), storageShapeB.size(), deviceDataB);
    aclTensor* c = aclCreateTensor(shapeB.data(), shapeB.size(), dtype2, stridesB.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   storageShapeB.data(), storageShapeB.size(), deviceDataB);
    aclTensor* result = aclCreateTensor(shapeResult.data(), shapeResult.size(), dtypeResult, stridesResult.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, storageShapeResult.data(), storageShapeResult.size(),
                                        deviceDataResult);

    auto inputs = OP_INPUT(a, b, c);
    auto outputs = OP_OUTPUT(result);
    op::internal::ProfilingInfoId profilingId;
    uint32_t opType = op::OpTypeDict::ToOpType("AddN");
    auto ctx = op::MakeOpArgContext(inputs, outputs);
    auto* launcher = new op::AiCoreKernelLauncher(opType, AI_CORE, profilingId, nullptr, ctx);

    auto bin = launcher->GetBin();
    ASSERT_EQ(bin, nullptr); // OP with dynamic input can only match bin if args use aclTensorList

    delete a;
    delete b;
    delete c;
    delete result;
    delete launcher;
}

TEST_F(TestBinaryReuse, AddnBinaryReuseSucc)
{
    cout << "test aclop AddN binaryReuse ..." << endl;
    CreateDynamicInputsAndCheckWithThreeInputs("AddN");
}

TEST_F(TestBinaryReuse, ConcatDBinaryBoolReuseSucc)
{
    cout << "test aclop ConcatD binaryReuse ..." << endl;
    CreateDynamicInputsAndCheckWithThreeInputs("ConcatD", ACL_BOOL);
}

TEST_F(TestBinaryReuse, AddnBinaryReuseSuccInput_20_float32)
{
    cout << "test aclop AddN binaryReuse input number = 20..." << endl;
    CreateAddNAndCheckWith20Inputs(ge::DT_FLOAT, ge::FORMAT_NC1HWC0);
}

TEST_F(TestBinaryReuse, AddnBinaryReuseSuccInput_20_float16)
{
    cout << "test aclop AddN binaryReuse input number = 20..." << endl;
    CreateAddNAndCheckWith20Inputs(ge::DT_FLOAT16, ge::FORMAT_NCDHW);
}

TEST_F(TestBinaryReuse, TransdataNchwTo5hdFp16)
{
    cout << "test transdata binaryReuse succ with format nchw -> 5hd fp16..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_88029d9b926ee3e9a5ce5a6d1528db14_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNchwTo5hdFp32)
{
    cout << "test transdata binaryReuse succ with format nchw -> 5hd fp32..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_8853381d7cfe72b1eab1b434f0aa9ba6_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNchwTo5hdDtypeNotMatch)
{
    cout << "test transdata binaryReuse succ with format nchw -> 5hd fp32->fp16. dtype not match..." << endl;
    string filePrefix = "";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, Transdata5hdToNchwFp32)
{
    cout << "test transdata binaryReuse succ with format 5hd -> nchw fp32 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_76ae9b382d5311eebe7ad615b3f47263_high_performance";
    CreateTransData(ge::FORMAT_NC1HWC0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, Transdata5hdToNchwFp16)
{
    cout << "test transdata binaryReuse succ with format 5hd -> nchw fp16 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_7251902a9905761d60ebb9db94b5d827_high_performance";
    CreateTransData(ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz01)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_cb5507159092f6eebe98693b30542645c16_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz02)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_70b0a60ea154b14167507ba38817355dc16_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_UINT32, ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, filePrefix);
}
TEST_F(TestBinaryReuse, TransdataNchwToFz03)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_9d79444575d50cda5471d978b574a58fc16_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_INT32, ge::FORMAT_FRACTAL_Z, ge::DT_INT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz04)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bf5f3735fb216678af8c8f035e8a3be5c32_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_INT8, ge::FORMAT_FRACTAL_Z, ge::DT_INT8, filePrefix);
}
TEST_F(TestBinaryReuse, TransdataNchwToFz05)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bf4b2fbdd8c73a05de981ea5b62a0ab3c32_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_UINT8, ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, filePrefix);
}
TEST_F(TestBinaryReuse, TransdataNchwToFz06)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_cb5507159092f6eebe98693b30542645c8_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz07)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_70b0a60ea154b14167507ba38817355dc8_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_UINT32, ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz08)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_9d79444575d50cda5471d978b574a58fc8_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_INT32, ge::FORMAT_FRACTAL_Z, ge::DT_INT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz09)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bf5f3735fb216678af8c8f035e8a3be5c32_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_INT8, ge::FORMAT_FRACTAL_Z, ge::DT_INT8, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataNchwToFz10)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bf4b2fbdd8c73a05de981ea5b62a0ab3c32_high_performance";
    CreateTransData(ge::FORMAT_NCHW, ge::DT_UINT8, ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz01)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bdb7c59a524cbe564249dda7625ef166c16_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz02)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_0c63936654c84a308f51ed03c8e0cbbac16_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_UINT32, ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz03)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a1532926856b5d4ea5bc081dfbe4dff3c16_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_INT32, ge::FORMAT_FRACTAL_Z, ge::DT_INT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz04)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_7dc3694ac20817642b666043dcf9d748c32_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_INT8, ge::FORMAT_FRACTAL_Z, ge::DT_INT8, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz05)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_5ca9788fbb7cc3d68aa798e541d1f3c4c32_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_UINT8, ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz06)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_bdb7c59a524cbe564249dda7625ef166c8_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz07)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_0c63936654c84a308f51ed03c8e0cbbac8_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_UINT32, ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz08)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a1532926856b5d4ea5bc081dfbe4dff3c8_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_INT32, ge::FORMAT_FRACTAL_Z, ge::DT_INT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz09)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_7dc3694ac20817642b666043dcf9d748c32_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_INT8, ge::FORMAT_FRACTAL_Z, ge::DT_INT8, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataHwcnToFz10)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_5ca9788fbb7cc3d68aa798e541d1f3c4c32_high_performance";
    CreateTransData(ge::FORMAT_HWCN, ge::DT_UINT8, ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataNchwToFzFp16)
{
    cout << "test transdata binaryReuse succ with format nchw -> fz fp16 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_c26ea265ede184f4042e734b3de29e12c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw01)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_45b1a39520df8e73d1b2f300a24777cac16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw02)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_3a79f329cb6d6c263ded8feb8455c167c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, ge::FORMAT_NCHW, ge::DT_UINT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw03)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_1b49b14e103ac2b5bb4434396dedca61c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT32, ge::FORMAT_NCHW, ge::DT_INT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw04)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_d826cfe4777ee0ea0fe05f15efca23ccc32_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT8, ge::FORMAT_NCHW, ge::DT_INT8, filePrefix);
}

// TEST_F(TestBinaryReuse, TransdataFzToNchw05AbnormalCase)
// {
//     string filePrefix = ""; // c0 will be set as 16, which should be 32 if we want to find the bin of uint8
//     CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, ge::FORMAT_NCHW,
//                     ge::DT_UINT8, filePrefix);
// }

TEST_F(TestBinaryReuse, TransdataFzToNchw06)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_45b1a39520df8e73d1b2f300a24777cac8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::DT_FLOAT, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw07)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_3a79f329cb6d6c263ded8feb8455c167c8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, ge::FORMAT_NCHW, ge::DT_UINT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw08)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_1b49b14e103ac2b5bb4434396dedca61c8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT32, ge::FORMAT_NCHW, ge::DT_INT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToNchw09)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_d826cfe4777ee0ea0fe05f15efca23ccc32_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT8, ge::FORMAT_NCHW, ge::DT_INT8, filePrefix, true);
}

// TEST_F(TestBinaryReuse, TransdataFzToNchw10AbnormalCase)
// {
//     string filePrefix = "";
//     CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, ge::FORMAT_NCHW,
//                     ge::DT_UINT8, filePrefix, true);
// }

TEST_F(TestBinaryReuse, TransdataFzToHwcn01)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_7b402173cd29f3f6826aa731c2958601c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, ge::FORMAT_HWCN, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn02)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_0728708c69496c73ad5913e3a62c7616c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, ge::FORMAT_HWCN, ge::DT_UINT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn03)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a37d5adda0f096ee0d5976e4f2baffb0c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT32, ge::FORMAT_HWCN, ge::DT_INT32, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn04)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a670e2d585fef91da69583fbb8a28d06c32_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT8, ge::FORMAT_HWCN, ge::DT_INT8, filePrefix);
}

// TEST_F(TestBinaryReuse, TransdataFzToHwcn05)
// {
//     string filePrefix = "";
//     CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, ge::FORMAT_HWCN,
//                     ge::DT_UINT8, filePrefix);
// }

TEST_F(TestBinaryReuse, TransdataFzToHwcn06)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_7b402173cd29f3f6826aa731c2958601c8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT, ge::FORMAT_HWCN, ge::DT_FLOAT, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn07)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_0728708c69496c73ad5913e3a62c7616c8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT32, ge::FORMAT_HWCN, ge::DT_UINT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn08)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a37d5adda0f096ee0d5976e4f2baffb0c8_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT32, ge::FORMAT_HWCN, ge::DT_INT32, filePrefix, true);
}

TEST_F(TestBinaryReuse, TransdataFzToHwcn09)
{
    string filePrefix = "kernel/ascend910/trans_data/TransData_a670e2d585fef91da69583fbb8a28d06c32_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_INT8, ge::FORMAT_HWCN, ge::DT_INT8, filePrefix, true);
}

// TEST_F(TestBinaryReuse, TransdataFzToHwcn10)
// {
//     string filePrefix = "";
//     CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_UINT8, ge::FORMAT_HWCN,
//                     ge::DT_UINT8, filePrefix, true);
// }

TEST_F(TestBinaryReuse, TransdataFzToNchwFp16)
{
    cout << "test transdata binaryReuse succ with format fz -> nchw fp16 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_c26ea265ede184f4042e734b3de29e12c16_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNdToNzFp32)
{
    cout << "test transdata binaryReuse succ with format nd -> nz fp32 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_67d7c40d127f4f68ac78cefc64cc7bd9_high_performance";
    CreateTransData(ge::FORMAT_ND, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNdToNzFp16)
{
    cout << "test transdata binaryReuse succ with format nd -> nz fp16 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_5d7204366c2b3d4caf8abb797ee26976_high_performance";
    CreateTransData(ge::FORMAT_ND, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, filePrefix);
}

// TEST_F(TestBinaryReuse, TransdataNdToNzFp16_with_c0_and_group)
// {
//
//     cout << "test transdata binaryReuse succ with format nd -> nz fp16 ..." << endl;
//     string filePrefix = "kernel/ascend910/trans_data/TransData_5d7204366c2b3d4caf8abb797ee26976_high_performance";
//     ge::Format formatIn = (ge::Format)ge::GetFormatFromSubAndC0(ge::FORMAT_ND, 1, 5);
//     ge::Format formatOut = (ge::Format)ge::GetFormatFromSubAndC0(ge::FORMAT_FRACTAL_NZ, 1, 5);
//     CreateTransData(formatIn, ge::DT_FLOAT16, formatOut,
//                     ge::DT_FLOAT16, filePrefix);
// }

TEST_F(TestBinaryReuse, TransdataNzToNdFp32)
{
    cout << "test transdata binaryReuse succ with format nz -> nd fp32 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_739be543d399b00434b6f682143e60c7_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT, ge::FORMAT_ND, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNzToNdFp16)
{
    cout << "test transdata binaryReuse succ with format nz -> nd fp16 ..." << endl;
    string filePrefix = "kernel/ascend910/trans_data/TransData_bdffbc0e3c487267fa44654652c07d54_high_performance";
    CreateTransData(ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, ge::FORMAT_ND, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNdToNdFp16FormatNotMatch)
{
    string filePrefix = "";
    CreateTransData(ge::FORMAT_ND, ge::DT_FLOAT, ge::FORMAT_ND, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, TransdataNzToNzFp16FormatNotMatch)
{
    cout << "test transdata binaryReuse succ with format nz -> nz fp16 ..." << endl;
    string filePrefix = "";
    CreateTransData(ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, Transdata_5hdToHwcnFp32_formatNotMatch)
{
    cout << "test transdata binaryReuse succ with format 5hd -> hwcn, format not match ..." << endl;
    string filePrefix = "";
    CreateTransData(ge::FORMAT_NC1HWC0, ge::DT_FLOAT, ge::FORMAT_HWCN, ge::DT_FLOAT, filePrefix);
}

TEST_F(TestBinaryReuse, MatMulNdInNdOutFp16AttrNotFound)
{
    string filePrefix = "";
    CreateMatMul(ge::FORMAT_ND, ge::DT_FLOAT16, ge::FORMAT_ND, ge::DT_FLOAT16, filePrefix);
}

TEST_F(TestBinaryReuse, MatMulNdInNdOutFp16WithAttrTrueFalse)
{
    string filePrefix = "kernel/ascend910/mat_mul/MatMulV2_ND_ND_FP16_FP16_true_false_all";
    CreateMatMul(ge::FORMAT_ND, ge::DT_FLOAT16, ge::FORMAT_ND, ge::DT_FLOAT16, filePrefix, true, true, false);
}

TEST_F(TestBinaryReuse, MatMulNdInNdOutFp16WithAttrFalseFalse)
{
    string filePrefix = "kernel/ascend910/mat_mul/MatMulV2_ND_ND_FP16_FP16_false_false_all";
    CreateMatMul(ge::FORMAT_ND, ge::DT_FLOAT16, ge::FORMAT_ND, ge::DT_FLOAT16, filePrefix, true, false, false);
}

TEST_F(TestBinaryReuse, MatMulDtypeUndefined)
{
    string filePrefix = "";
    CreateMatMul(ge::FORMAT_ND, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::DT_FLOAT16, filePrefix, true, false, false);
}

TEST_F(TestBinaryReuse, ConcatDInt16AndFloat16)
{
    string filePrefix = "kernel/ascend910/concat_d/ConcatD_8460c4b727ab5cc93a9595333a3fd571_high_performance";
    CreateConcatD(aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_NCHW, filePrefix);

    CreateConcatD(aclDataType::ACL_INT16, aclFormat::ACL_FORMAT_ND, filePrefix);
}

TEST_F(TestBinaryReuse, ConcatDInt8AndUint8)
{
    string filePrefix = "kernel/ascend910/concat_d/ConcatD_022c2ffc1fc8b7b7fce31a48ed50ca2c_high_performance";
    CreateConcatD(aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_NCHW, filePrefix);

    CreateConcatD(aclDataType::ACL_UINT8, aclFormat::ACL_FORMAT_ND, filePrefix);
}

TEST_F(TestBinaryReuse, ConcatDInt32AndFloat)
{
    string filePrefix = "kernel/ascend910/concat_d/ConcatD_3c918d330f983e55e90427727bb63ca4_high_performance";
    CreateConcatD(aclDataType::ACL_INT32, aclFormat::ACL_FORMAT_NCHW, filePrefix);

    CreateConcatD(aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, filePrefix);
}

TEST_F(TestBinaryReuse, ConcatDDoubleAndInt64)
{
    string filePrefix = "kernel/ascend910/concat_d/ConcatD_7cd4f521247dc2e837d3befaa0c89373_high_performance";
    CreateConcatD(aclDataType::ACL_DOUBLE, aclFormat::ACL_FORMAT_ND, filePrefix);

    CreateConcatD(aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, filePrefix);
}

TEST_F(TestBinaryReuse, LogFloat16AndFloat)
{
    string filePrefix = "kernel/ascend910/log/Log_b9e1030977b1dc266a4781fe63584cae_high_performance";
    CreateSingleInAndout("Log", ge::DT_FLOAT16, ge::FORMAT_NCHW, filePrefix, -1.0, 1.0, 0.0);
    filePrefix = "kernel/ascend910/log/Log_fb42ea349cd1f1779279250bf66d5d90_high_performance";
    CreateSingleInAndout("Log", ge::DT_FLOAT, ge::FORMAT_ND, filePrefix, 2.0, 1.0, 0.0);
    filePrefix = "";
    CreateSingleInAndout("Log", ge::DT_FLOAT, ge::FORMAT_ND, filePrefix, 3.0, 1.0, 0.0);
}

TEST_F(TestBinaryReuse, GeluFloat16AndFLoat32UseHighPrecision)
{
    string filePrefix = "kernel/ascend910/gelu/Gelu_0_high_precision";
    CreateSingleInAndout("Gelu", ge::DT_FLOAT16, ge::FORMAT_NCHW, filePrefix, 1.0);
    filePrefix = "kernel/ascend910/gelu/Gelu_1_high_precision";
    CreateSingleInAndout("Gelu", ge::DT_FLOAT, ge::FORMAT_ND, filePrefix, 1);
}

TEST_F(TestBinaryReuse, Conv2DBackPropFilterDataFormatNCHW)
{
    string filePrefix = "kernel/ascend910/conv2d_backprop_input/Conv2DBackpropInput_NC1HWC0_NC1HWC0_FP16_FP16_NCHW_all";
    const char* nchw = "NCHW";
    std::string nchwStr("NCHW");
    std::string nhwcStr("NHWC");
    int64_t value = 1;
    aclIntArray strides(&value, 1);
    aclIntArray pads(&value, 1);
    aclIntArray dilations(&value, 1);

    CreateConv2DBackpropInput(filePrefix, &strides, &pads, &dilations, value, nchw);
    CreateConv2DBackpropInput(filePrefix, &strides, &pads, &dilations, 5, nchwStr);
    CreateConv2DBackpropInput(filePrefix, &strides, &pads, &dilations, 0, nchw, &strides, &pads, 0, nchw);

    filePrefix = "";
    CreateConv2DBackpropInput(filePrefix, &strides, &pads, &dilations, 0, nhwcStr);
}

TEST_F(TestBinaryReuse, ReduceAllFloat16AndFLoat32UseHighPrecision)
{
    string filePrefix = "kernel/ascend910/reduce_all/ReduceAll_a9b7af94359fc7c19da07ad0790f065c_high_performance";
    CreateTwoInAndSingleout("ReduceAll", ge::DT_INT8, ge::DT_INT32, ge::FORMAT_NCHW, filePrefix);
    filePrefix = "kernel/ascend910/reduce_all/ReduceAll_bbb33f17aca2fbbf645166c6cba610fc_high_performance";
    CreateTwoInAndSingleout("ReduceAll", ge::DT_INT8, ge::DT_INT64, ge::FORMAT_ND, filePrefix);
}

TEST_F(TestBinaryReuse, TestStaticBinaryAdd)
{
    string
        filePrefix = "static_kernel/ai_core/static_kernel_202307261051/TestStaticAdd/TestStaticAdd_high_performance_0";
    TestStaticBinaryAdd(filePrefix);
}

TEST_F(TestBinaryReuse, TopKV2)
{
    string filePrefix = "kernel/ascend910/top_k_v2/TopKV2_5cc1ed45ef52d7db0697fa437556176a_high_performance";
    CreateTopKV2("TopKV2", aclDataType::ACL_FLOAT16, aclDataType::ACL_INT32, aclFormat::ACL_FORMAT_NCHW, filePrefix);
}

TEST_F(TestBinaryReuse, NLLLoss)
{
    string filePrefix = "kernel/ascend910/nll_loss/NLLLoss_cb871abe297529940e37590528e15ab2_high_performance";
    CreateNLLLoss("NLLLoss", aclDataType::ACL_FLOAT, aclDataType::ACL_INT32, aclFormat::ACL_FORMAT_NCHW, 2, filePrefix,
                  "mean");

    filePrefix = "kernel/ascend910/nll_loss/NLLLoss_193d89f5bc0a57ed7aa7da64cb1464e6_high_performance";
    CreateNLLLoss("NLLLoss", aclDataType::ACL_FLOAT, aclDataType::ACL_INT32, aclFormat::ACL_FORMAT_NCHW, 3, filePrefix,
                  "none");
}

TEST_F(TestBinaryReuse, Conv2DBackpropFilter)
{
    string
        filePrefix = "kernel/ascend910/conv2d_backprop_filter/Conv2DBackpropFilter_NC1HWC0_FRACTALZ_FP16_FP32_NCHW_all";
    CreateConv2DBackpropFilter(filePrefix);
}

TEST_F(TestBinaryReuse, LpNormUpdate)
{
    string
        filePrefix = "kernel/ascend910/lp_norm_update/LpNormUpdate_8e56095e46d7eb72c220743f32c6de6c_high_performance";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 0, 0.0);

    filePrefix = "kernel/ascend910/lp_norm_update/LpNormUpdate_0f8e56e6ec4f4f861ce1e4d6e60f6c4a_high_performance";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 0, 1e-12);

    filePrefix = "";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 1, 1.0);

    filePrefix = "";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 4, 0.0);

    filePrefix = "kernel/ascend910/lp_norm_update/LpNormUpdate_b718a1eaea6b8a3f44c2f37bd05db2d5_high_performance";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 3, 0.0);

    filePrefix = "";
    CreateSingleInAndout("LpNormUpdate", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 3, 0.0000001);
}

TEST_F(TestBinaryReuse, Renorm)
{
    string filePrefix = "kernel/ascend910/renorm/Renorm_49ebb7bc1057f23132a146629c814876_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 1.0);

    filePrefix = "kernel/ascend910/renorm/Renorm_e9fd33eb0c7c94efc12f1216acc753dc_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 2.0);

    float a = 2.0;
    filePrefix = "kernel/ascend910/renorm/Renorm_e9fd33eb0c7c94efc12f1216acc753dc_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, a);

    const float b = 2.0;
    filePrefix = "kernel/ascend910/renorm/Renorm_e9fd33eb0c7c94efc12f1216acc753dc_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, b);

    filePrefix = "";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 2.1);

    filePrefix = "";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 5.1);

    filePrefix = "kernel/ascend910/renorm/Renorm_e9fd33eb0c7c94efc12f1216acc753dc_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 2.00000);

    filePrefix = "kernel/ascend910/renorm/Renorm_8d3d70c081beb620647856890a07a09c_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 3.0000000);

    filePrefix = "kernel/ascend910/renorm/Renorm_a2affea3615b02951dbf601e103ed96a_high_performance";
    CreateSingleInAndout("Renorm", ge::DT_FLOAT, ge::FORMAT_NCHW, filePrefix, 0.0000000);
}

TEST_F(TestBinaryReuse, SimplifiedKeyTest)
{
    string filePrefix = "kernel/ascend910/simplifiedkeytest/"
                        "Simplifiedkeytest_8bed43affdbf596d834d2a0a046ee716_high_performance";
    CreateSingleInAndout("Simplifiedkeytest", ge::DT_FLOAT16, ge::FORMAT_ND, "", 1.0);
}

TEST_F(TestBinaryReuse, Memset)
{
    auto unique_executor = CREATE_EXECUTOR();
    string filePrefix = "";
    CreateNoInputOutput("MemSet", filePrefix, 1);
    int64_t sizes[1] = {5};
    aclIntArray* sizesArray = unique_executor->AllocIntArray(sizes, 1);
    filePrefix = "kernel/ascend910/mem_set/MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArray);

    int64_t sizesTwo[2] = {65, 12};
    aclIntArray* sizesArrayTwo = unique_executor->AllocIntArray(sizesTwo, 2);
    filePrefix = "kernel/ascend910/mem_set/MemSet_e0d52ae24534213ca40deeb3ccd89c06_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArrayTwo);

    const int64_t sizesTwoConst[2] = {65, 12};
    const aclIntArray* sizesArrayTwoConst = unique_executor->AllocIntArray(sizesTwoConst, 2);
    filePrefix = "kernel/ascend910/mem_set/MemSet_e0d52ae24534213ca40deeb3ccd89c06_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArrayTwoConst);

    //  filePrefix = "kernel/ascend910/mem_set/MemSet_e0d52ae24534213ca40deeb3ccd89c06_high_performance";
    //  CreateNoInputOutput("MemSet", filePrefix, sizesTwoConst);
}

TEST_F(TestBinaryReuse, LayerNormBetaGammaBackpropV2)
{
    auto unique_executor = CREATE_EXECUTOR();
    int64_t sizesOne[1] = {65};
    aclIntArray* sizesArrayOne = unique_executor->AllocIntArray(sizesOne, 1);

    int64_t sizesTwo[2] = {65, 12};
    aclIntArray* sizesArrayTwo = unique_executor->AllocIntArray(sizesTwo, 2);

    string filePrefix = "kernel/ascend910/layer_norm_beta_gamma_backprop_v2/"
                        "LayerNormBetaGammaBackpropV2_ee802680189c60728cbb190ec5b2cbae_high_performance";
    CreateLayerNormBetaGammaBackpropV2(filePrefix, sizesArrayOne, sizesArrayTwo);

    filePrefix = "kernel/ascend910/layer_norm_beta_gamma_backprop_v2/"
                 "LayerNormBetaGammaBackpropV2_376858b1dea114e8dc152045b5f9f4ff_high_performance";
    CreateLayerNormBetaGammaBackpropV2(filePrefix, sizesArrayOne, sizesArrayOne);

    filePrefix = "";
    CreateLayerNormBetaGammaBackpropV2(filePrefix, sizesArrayOne);
}

TEST_F(TestBinaryReuse, TestNonZero)
{
    string filePrefix = "kernel/ascend910/non_zero/NonZero_edb06daebcc6f079a6dee9a7ed831ea9_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2}, {2, 3, 4, 5}, filePrefix,
                                  true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_6e5a6cd9de928cb056e28cda78fe447e_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3}, {2, 3, 4, 5},
                                  filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_1267d067bf644ff538330671fcfcfd9f_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4}, {2, 3, 4, 5},
                                  filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_27a478f4fafa16a7537521a403837911_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4, 5}, {2, 3, 4, 5},
                                  filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_91d091fdf3121283326bbfe670fa793b_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4, 5, 6}, {2, 3, 4, 5},
                                  filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_811ba726116e13ff3bd4608c72e11076_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4, 5, 6, 7},
                                  {2, 3, 4, 5}, filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_d399326088830ff05042304adc992e13_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4, 5, 6, 7, 8},
                                  {2, 3, 4, 5}, filePrefix, true);

    filePrefix = "kernel/ascend910/non_zero/NonZero_e06011186220d191c725cbb04059b324_high_performance";
    CreateSingleInAndoutWithShape("NonZero", ge::DT_BOOL, ge::DT_INT32, ge::FORMAT_NCHW, {2, 3, 4, 5, 6, 7, 8, 9},
                                  {2, 3, 4, 5}, filePrefix, true);
}
// TEST_F(TestBinaryReuse, RenormAbnormalCase) {
//   string filePrefix = "kernel/ascend910/renorm/Renorm_49ebb7bc1057f23132a146629c814876_high_performance";
//   const char* invalidStr = "IIIIIIIIIIIVVVVVVVVVVVAAAAAAAAALLLLLLLLLLIIIIIIIIIIDDDDDDDDD";
//   CreateSingleInAndout("Renorm", aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW, filePrefix,
//                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, invalidStr);
// }
//
// TEST_F(TestBinaryReuse, Conv2DBackPropFilterAbnormalCase) {
//   string filePrefix =
//   "kernel/ascend910/conv2d_backprop_input/Conv2DBackpropInput_NC1HWC0_NC1HWC0_FP16_FP16_NCHW_all"; const char
//   *abnormalNchw =
//   "NCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHWNCHW";
//   int64_t value = 1;
//   aclIntArray strides(&value, 1);
//   aclIntArray pads(&value, 1);
//   aclIntArray dilations(&value, 1);
//
//   CreateConv2DBackpropInput(filePrefix, &strides, &pads, &dilations, 0, abnormalNchw, &strides, &pads, 0,
//   abnormalNchw);

//}
