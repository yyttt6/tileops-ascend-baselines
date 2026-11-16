/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_stub.h"
#include "string.h"
#include "register/op_impl_registry.h"
#include "executor/indv_bininfo.h"
#include "utils/indv_base.h"
#include "executor/indv_collector.h"
#include "register/tilingdata_base.h"
#include <assert.h>

using namespace optiling;
#ifdef __cplusplus
extern "C" {
#endif

void NnopbaseGetDimsFromGertTensor(GertTensor* tensor, int64_t* dims, size_t* num)
{
    size_t dimNum;
    GertShape shape = tensor->GetOriginShape();
    dimNum = shape.GetDimNum();
    OP_LOGI("NnopbaseGetDimsFromGertTensor dimNum %d \n", dimNum);
    for (size_t j = 0; j < dimNum; j++) {
        OP_LOGI("NnopbaseGetDimsFromGertTensor shape.GetDim(j) %d \n", shape.GetDim(j));
        dims[j] = shape.GetDim(j);
    }
    *num = dimNum;
}

struct BNInferenceDParam {
    int dim0[5], dim1[5], totalSize, blockFactor, ubFactor;
    u_int32_t factorDataLen{61440}, factorParamLen{2048};
};

#define min(a, b) ((a) > (b) ? (b) : (a))
void TilingParamBn(BNInferenceDParam& param, size_t dimNum, int64_t* dims, int blockDim)
{
    int n, c1, h, w, c0;
    n = (int)dims[0];
    c1 = (int)dims[1];
    h = (int)dims[2];
    w = (int)dims[3];
    c0 = (int)dims[4];

    OP_LOGI("TilingParamBn %d %d %d %d %d\n", n, c1, h, w, c0);
    const int64_t totalSize = n * c1 * h * w * c0;
    int64_t blockFactor = (totalSize + blockDim - 1) / blockDim;
    int dim0[5] = {n, c1, h, w, c0};
    int dim1[5] = {1, c1, 1, 1, c0};
    for (int i = 0; i < 5; i++) {
        param.dim0[i] = dim0[i];
        param.dim1[i] = dim1[i];
    }
    param.totalSize = totalSize;
    param.blockFactor = blockFactor;
    param.factorParamLen = c1 * c0 * 2 * 2;
    const auto hwc0 = dim0[2] * dim0[3] * dim0[4];
    const auto computeCount = param.factorDataLen / 2; // 30720
    const auto factorCount = (hwc0 + computeCount - 1) / computeCount;
    unsigned int hwCount = 2;
    while (true) {
        if (factorCount <= hwCount) {
            break;
        }
        hwCount *= 2;
    }
    param.ubFactor = ((unsigned int)hwc0 <= computeCount) ?
                         min(((unsigned int)param.blockFactor), computeCount / hwc0 * hwc0) :
                         hwc0 / hwCount;
}

int TilingBnStub(unsigned char* buf, int len, GertTensor* tensor, int64_t* batchsz, uint64_t* tilingKey,
                 uint32_t* blockDim)
{
    BNInferenceDParam* param;
    int64_t dims[20];
    size_t num = 0;

    if ((unsigned int)len < sizeof(BNInferenceDParam)) {
        return 0;
    }
    param = (BNInferenceDParam*)buf;
    param->factorDataLen = 61440;
    param->factorParamLen = 2048;

    NnopbaseGetDimsFromGertTensor(tensor, dims, &num);
    TilingParamBn(*param, num, dims, 8);

    *batchsz = 0;
    *tilingKey = 1;
    *blockDim = 8;
    return sizeof(BNInferenceDParam);
};

unsigned int TilingForBn(gert::TilingContext* context_)
{
    KernelRunContext* context = reinterpret_cast<KernelRunContext*>(context_);
    int64_t batchsz = 0;
    uint64_t* tilingKey = (uint64_t*)(context->output_start[0]->data.inplace);
    uint32_t* blockDim = (uint32_t*)(context->output_start[1]->data.inplace);
    bool* atomicClean = (bool*)(context->output_start[2]->data.inplace);                            // 2 for output;
    NnopbaseTilingData* tilingData = (NnopbaseTilingData*)(context->output_start[3]->data.pointer); // 3 for tiling data
    NnopbaseWorkspaceSizes* ws = (NnopbaseWorkspaceSizes*)(context->output_start[4]->data.pointer); // 4 for workspace
    GertTensor* tensor = nullptr;
    if (context->input_size > 0) {
        tensor = (GertTensor*)context->values[0]->data.pointer;
    }
    tilingData->SetDataSize(TilingBnStub((unsigned char*)tilingData->GetData(), tilingData->GetCapacity(), tensor,
                                         &batchsz, tilingKey, blockDim));
    uint32_t* aicpuBlockDim = (uint32_t*)(context->output_start[8]->data.inplace);
    *aicpuBlockDim = 2U;
    *atomicClean = false;
    return 0;
}

// BEGIN_TILING_DATA_DEF(AddTik2TilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, blockDim);
//   TILING_DATA_FIELD_DEF(uint32_t, totalLength);
//   TILING_DATA_FIELD_DEF(uint32_t, tileNum);
// END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(AddTik2, AddTik2TilingData)

// unsigned int TilingForAdd(gert::TilingContext* context)
// {
//     AddTik2TilingData tiling;
//     uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
//     context->SetBlockDim(8);
//     tiling.set_blockDim(8);
//     tiling.set_totalLength(totalLength);
//     tiling.set_tileNum(8);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//     context->SetTilingKey(1);
//     size_t sysWorkspaceSize = 100;
//     size_t *currentWorkspace = context->GetWorkspaceSizes(2);
//     if (context->GetWorkspaceNum() <= 0) {
//         std::cout << "GetWorkspaceNum Failed"<<std::endl;
//         return ge::GRAPH_FAILED;
//     }
//     currentWorkspace[0] = sysWorkspaceSize;
//     currentWorkspace[1] = sysWorkspaceSize;
//     auto *tensor1 = context->GetOptionalInputTensor(2);
//     if (tensor1 != nullptr) {
//         auto tensor1_data = tensor1->GetData<float>();
//         std::cout << "tensor1_data[0] is " << tensor1_data <<std::endl;
//         assert(tensor1_data[0] == 2.0);
//     }
//     auto *tensor2 = context->GetOptionalInputTensor(3);
//     if (tensor2 != nullptr) {
//         auto tensor2_data = tensor2->GetData<int64_t>();
//         std::cout << "tensor2_data[0] is " << tensor2_data <<std::endl;
//         assert(tensor2_data[0] == 3);
//     }
//     auto *tensor3 = context->GetOptionalInputTensor(4);
//     if (tensor3 != nullptr) {
//         auto tensor3_data = tensor3->GetData<bool>();
//         std::cout << "tensor3_data[0] is " << tensor3_data <<std::endl;
//         assert(tensor3_data[0]);
//     }
//     return 0;
// }

// BEGIN_TILING_DATA_DEF(custom_op1TilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, blockDim);
//   TILING_DATA_FIELD_DEF(uint32_t, totalLength);
//   TILING_DATA_FIELD_DEF(uint32_t, tileNum);
// END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(custom_op1, custom_op1TilingData)

// unsigned int TilingForCustomOp1(gert::TilingContext* context)
// {
//     custom_op1TilingData tiling;
//     uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
//     context->SetBlockDim(8);
//     tiling.set_blockDim(8);
//     tiling.set_totalLength(totalLength);
//     tiling.set_tileNum(8);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//     context->SetTilingKey(1);
//     size_t sysWorkspaceSize = 100;
//     size_t *currentWorkspace = context->GetWorkspaceSizes(2);
//     if (context->GetWorkspaceNum() <= 0) {
//         std::cout << "GetWorkspaceNum Failed"<<std::endl;
//         return ge::GRAPH_FAILED;
//     }
//     currentWorkspace[0] = sysWorkspaceSize;
//     currentWorkspace[1] = sysWorkspaceSize;
//     return 0;
// }

// BEGIN_TILING_DATA_DEF(custom_op2TilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, blockDim);
//   TILING_DATA_FIELD_DEF(uint32_t, totalLength);
//   TILING_DATA_FIELD_DEF(uint32_t, tileNum);
// END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(custom_op2, custom_op2TilingData)

// unsigned int TilingForCustomOp2(gert::TilingContext* context)
// {
//     custom_op2TilingData tiling;
//     uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
//     context->SetBlockDim(8);
//     tiling.set_blockDim(8);
//     tiling.set_totalLength(totalLength);
//     tiling.set_tileNum(8);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//     context->SetTilingKey(1);
//     size_t sysWorkspaceSize = 100;
//     size_t *currentWorkspace = context->GetWorkspaceSizes(2);
//     if (context->GetWorkspaceNum() <= 0) {
//         std::cout << "GetWorkspaceNum Failed"<<std::endl;
//         return ge::GRAPH_FAILED;
//     }
//     currentWorkspace[0] = sysWorkspaceSize;
//     currentWorkspace[1] = sysWorkspaceSize;
//     return 0;
// }

// BEGIN_TILING_DATA_DEF(NonFiniteCheckTilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, blockDim);
//   TILING_DATA_FIELD_DEF(uint32_t, totalLength);
//   TILING_DATA_FIELD_DEF(uint32_t, tileNum);
// END_TILING_DATA_DEF;

// unsigned int TilingForNonFiniteCheck(gert::TilingContext* context)
// {
//     NonFiniteCheckTilingData tiling;
//     uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
//     context->SetBlockDim(8);
//     tiling.set_blockDim(8);
//     tiling.set_totalLength(totalLength);
//     tiling.set_tileNum(8);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//     context->SetTilingKey(1);
//     size_t *currentWorkspace = context->GetWorkspaceSizes(1);
//     currentWorkspace[0] = 100;
//     return 0;
// }

#ifdef __cplusplus
}
#endif
