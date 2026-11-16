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

#include "op_ctx_def.h"
#include "op_run_context.h"
#include "kernel_launcher.h"
#include "opdev/make_op_executor.h"
#include "utils/file_faker.h"
#include "thread_local_context.h"

using namespace op;
using namespace std;
// OP_TYPE_REGISTER(TestStaticAdd)
// OP_TYPE_REGISTER(Conv2D)
class TestDebugKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::internal::systemConfig.SetEnableDebugKernelFlag(true);
        op::GenOpTypeId("AddN");
        op::GenOpTypeId("Conv2D");
        op::GenOpTypeId("ViewCopy");
        op::GenOpTypeId("MemSet");
        op::GenOpTypeId("TestStaticAdd");
        // Log is to test invalid debug kernel config json file
        op::GenOpTypeId("Log");
        for (size_t i = 1; i < OpTypeDict::GetAllOpTypeSize(); ++i) {
            op::internal::KernelMgr::getInstance().AclOpKernelInit(i);
        }
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
        std::cout << "TestDebugKernel SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::internal::systemConfig.SetEnableDebugKernelFlag(false);
        std::cout << "TestDebugKernel TearDown" << std::endl;
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

    void CheckResult(op::internal::OpKernelBin* bin, bool debug, const string& filePrefix)
    {
        std::string prefix = "ascend910";
        if (filePrefix.empty()) {
            ASSERT_EQ(bin, nullptr);
        } else {
            std::string jsonPath = filePrefix + ".json";
            std::string binPath = filePrefix + ".o";

            ASSERT_NE(bin, nullptr);

            auto debugPos = bin->binPath_.find("debug_kernel");
            if (debug) {
                bool res = debugPos != string::npos;
            } else {
                bool res = debugPos == string::npos;
            }

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

#define LAUNCH_AND_CHECK(opTypeId, CORE_TYPE, profilingInfoId, filePrefix, debug, op_args...)              \
    do {                                                                                                   \
        auto ctx = op::MakeOpArgContext(op_args);                                                          \
        auto* launcher = new op::AiCoreKernelLauncher{opTypeId, CORE_TYPE, profilingInfoId, nullptr, ctx}; \
        auto bin = launcher->GetBin();                                                                     \
        CheckResult(bin, debug, filePrefix);                                                               \
        delete launcher;                                                                                   \
    } while (0)

    void CreateDynamicInputsAndCheckWithThreeInputs(const string& opTypeStr, aclDataType dataType = ACL_INT32)
    {
        op::DataType dtype = static_cast<op::DataType>(dataType);
        auto unique_executor = CREATE_EXECUTOR();
        aclTensor* tensors[2] = {nullptr};
        for (auto& tensor : tensors) {
            tensor = unique_executor->AllocTensor({1, 6, 8}, dtype, ge::FORMAT_ND);
        }

        aclTensorList* tensorList = unique_executor->AllocTensorList(reinterpret_cast<const aclTensor**>(&tensors), 2);
        aclTensor* result = unique_executor->AllocTensor({1, 6, 8}, dtype, ge::FORMAT_ND);

        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(opTypeStr);
        string filePrefix = "ascend910/add_n/AddN_8b4ec10fbb39c5c0a65192c606400b29_high_performance";

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, /*debug=*/true, OP_INPUT(tensorList),
                         OP_OUTPUT(result));

        auto ctx = op::MakeOpArgContext(OP_INPUT(tensorList), OP_OUTPUT(result));
        auto* launcher = new op::AiCoreKernelLauncher{opType, AI_CORE, profilingId, nullptr, ctx};
        auto bin = launcher->GetBin();

        auto inputTuple = OP_INPUT(tensorList);
        auto outputTuple = OP_OUTPUT(result);
        auto attrTuple = OP_ATTR();
        auto ws = OP_WORKSPACE();
        auto ctx2 = op::MakeOpArgContext(inputTuple, outputTuple, attrTuple, ws);
        int dummyStream = 0;
        void* stream = &dummyStream;
        bool original_flag = op::internal::RtsApiFlag::GetRtsApiFlag().IfNewApi();
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(false);
        auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx2);
        op::DestroyOpArgContext(ctx2);
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(original_flag);
        EXPECT_EQ(rc, ACLNN_SUCCESS);

        EXPECT_EQ(bin->oriOpParaSize_, 128);
        EXPECT_EQ(bin->debugConfig_, 63);

        auto tilingRes = op::internal::OpRunContextMgr::opRunCtx_.tilingCtx_.GetTilingResult();
        EXPECT_NE(tilingRes, nullptr);
        void* dfxInfoAtomicIndex = op::internal::PtrShift(tilingRes->tilingData_->data_,
                                                          tilingRes->tilingData_->data_size_ - sizeof(uint64_t));
        EXPECT_EQ(*static_cast<uint64_t*>(dfxInfoAtomicIndex), 5525);
        for (int i = 2; i <= 4; i++) {
            void* shapeSizePtr = op::internal::PtrShift(tilingRes->tilingData_->data_,
                                                        tilingRes->tilingData_->data_size_ - i * sizeof(uint64_t));
            std::cout << *static_cast<uint64_t*>(shapeSizePtr) << std::endl;
            EXPECT_EQ(*static_cast<uint64_t*>(shapeSizePtr), 192);
        }
        delete launcher;
    }

    template <typename... ATTRS>
    void CreateNoInputOutput(const string& op_type, const string& filePrefix, ATTRS&&... opAttrs)
    {
        op::internal::ProfilingInfoId profilingId;
        uint32_t opType = op::OpTypeDict::ToOpType(op_type);
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, /*debug=*/false,
                         OP_ATTR(std::forward<ATTRS>(opAttrs)...));
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
        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, /*debug=*/false, OP_INPUT(input0, input1),
                         OP_OUTPUT(output));
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

        LAUNCH_AND_CHECK(opType, AI_CORE, profilingId, filePrefix, /*debug=*/true, OP_INPUT(input0, input1),
                         OP_OUTPUT(output0),
                         OP_ATTR(stridesIntArray, paddingsIntArray, dilationsIntArray, groups, dataFormat, offset_x));
    }
};

// TEMP_COMMENT
// TEST_F(TestDebugKernel, AddnBinaryReuseSucc)
// {
//     cout << "test aclop AddN binaryReuse ..." << endl;
//     CreateDynamicInputsAndCheckWithThreeInputs("AddN");
// }

TEST_F(TestDebugKernel, TestStaticBinaryConv2D)
{
    string filePrefix = "debug_kernel/ascend910/Conv2D_high_performance_0";
    TestStaticBinaryConv2D(filePrefix);
}

TEST_F(TestDebugKernel, TestStaticBinaryAdd)
{
    string
        filePrefix = "static_kernel/ai_core/static_kernel_202307261051/TestStaticAdd/TestStaticAdd_high_performance_0";
    TestStaticBinaryAdd(filePrefix);
}

TEST_F(TestDebugKernel, TestOpMemset)
{
    auto unique_executor = CREATE_EXECUTOR();
    string filePrefix = "";
    CreateNoInputOutput("MemSet", filePrefix, 1);
    int64_t sizes[1] = {5};
    aclIntArray* sizesArray = unique_executor->AllocIntArray(sizes, 1);
    filePrefix = "ascend910/mem_set/MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArray);

    int64_t sizesTwo[2] = {65, 12};
    aclIntArray* sizesArrayTwo = unique_executor->AllocIntArray(sizesTwo, 2);
    filePrefix = "ascend910/mem_set/MemSet_e0d52ae24534213ca40deeb3ccd89c06_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArrayTwo);

    const int64_t sizesTwoConst[2] = {65, 12};
    const aclIntArray* sizesArrayTwoConst = unique_executor->AllocIntArray(sizesTwoConst, 2);
    filePrefix = "ascend910/mem_set/MemSet_e0d52ae24534213ca40deeb3ccd89c06_high_performance";
    CreateNoInputOutput("MemSet", filePrefix, sizesArrayTwoConst);
}
