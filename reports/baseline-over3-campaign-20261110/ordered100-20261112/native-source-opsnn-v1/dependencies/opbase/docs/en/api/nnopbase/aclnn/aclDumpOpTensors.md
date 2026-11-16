# aclDumpOpTensors

## Function

During model execution, the input and output tensor data of operators can be dumped. This helps locate and analyze operator input and output data issues.

## Prototype

```cpp
aclnnStatus aclDumpOpTensors(const char *opType, const char *opName, aclTensor **tensors, size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| opType | Input| A string indicating the operator type, for example, `Add`.|
| opName | Input| A string indicating the operator name, for example, `add_custom`.|
| tensors | Input| 1D tensor, indicating the pointers to the input/output tensors to be dumped. Note the tensor sequence. Input tensors must precede output tensors.|
| inputTensorNum | Input| Number of input tensors to be dumped.|
| outputTensorNum | Input| Number of output tensors to be dumped.|
| stream | Input| The execution stream where the dump occurs.|

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

This API takes effect only when the operator dump function is enabled. You can enable this function by calling the aclInit API or the aclmdlInitDump, aclmdlSetDump, and aclmdlFinalizeDump series APIs. For details about the APIs, see Runtime APIs.

## Examples

The following code examples are for reference only and are not intended for direct copying and execution:

1. Enable the operator dump function by calling the aclInit API. See the following code snippet:

    ```cpp
    // Initialize resources.
    aclInit("./acl.json");
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ```

    The following is an example of the acl.json file. For details, see the model dump configuration and single-operator dump configuration examples in the aclInit API Reference.

    ```json
    {
        "dump": {
            "dump_path": "./",
            "dump_list": [],
            "dump_mode": "all",
            "dump_data": "tensor"
        }
    }
    ```

2. Refer to the key pseudocode for calling this API as follows (using the torch operator as an example):

    ```cpp
    #include <torch/extension.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/core/npu/NPUFunctions.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
    #include "torch_npu/csrc/core/npu/register/OptionsManager.h"
    #include "torch_npu/csrc/aten/NPUNativeFunctions.h"
    #include "torch_npu/csrc/flopcount/FlopCount.h"
    #include "torch_npu/csrc/flopcount/FlopCounter.h"
    #include "torch_npu/csrc/core/npu/NpuVariables.h"
    #include "kernel_operator.h"
    #include <acl/acl_base.h>
    #include <aclnn/acl_meta.h>

    constexpr int32_t BUFFER_NUM = 2;
    constexpr int64_t MAX_DIM_NUM = 5;
    constexpr int64_t NCL_DIM_NUM = 3;
    constexpr int64_t NCHW_DIM_NUM = 4;
    constexpr int64_t NCDHW_DIM_NUM = 5;

    // Generate the 1D tensor of the input/output tensor object pointers of the operator to be dumped.
    #define INIT_ACL_TENSOR_ARRAY(tensors, ...) aclTensor* tensors[] = {__VA_ARGS__}

    // Convert the at::Tensor object into an aclTensor object. This function simplifies the processing. Configure the parameters based on the actual operator.
    aclTensor *ConvertTensor(const at::Tensor &at_tensor)
    {
        aclDataType acl_data_type = ACL_FLOAT16;
        c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;

        const auto dimNum = at_tensor.sizes().size();
        aclFormat format = ACL_FORMAT_ND;
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }

        auto acl_tensor =
            aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                            at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                            const_cast<void *>(at_tensor.storage().data()));
        return acl_tensor;
    }

    // Custom operator implementation. Configure the parameters based on the actual operator.
    class KernelAdd {
    public:
        __aicore__ inline KernelAdd() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
        {
            this->blockLength = totalLength / AscendC::GetBlockNum();
            this->tileNum = 8;
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
        }
        __aicore__ inline void Process()
        {
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress)
        {
            AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
            AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
            AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }
        __aicore__ inline void Compute(int32_t progress)
        {
            AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
            AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
            AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
            AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
            outQueueZ.EnQue<half>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress)
        {
            AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
            AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
            outQueueZ.FreeTensor(zLocal);
        }

    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
        AscendC::GlobalTensor<half> xGm;
        AscendC::GlobalTensor<half> yGm;
        AscendC::GlobalTensor<half> zGm;
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
    };

    __global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        KernelAdd op;
        op.Init(x, y, z, totalLength);
        op.Process();
    }

    namespace ascendc_ops {
    at::Tensor ascendc_add(const at::Tensor& x, const at::Tensor& y)
    {
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        uint32_t numBlocks = 8;
        uint32_t totalLength = 1;
        for (uint32_t size : x.sizes()) {
            totalLength *= size;
        }

        add_custom<<<numBlocks, nullptr, aclStream>>>((uint8_t*)(x.mutable_data_ptr()), (uint8_t*)(y.mutable_data_ptr()), (uint8_t*)(z.mutable_data_ptr()), totalLength);

        // Dump the input and output tensor data of the operator.
        INIT_ACL_TENSOR_ARRAY(tensors, ConvertTensor(x), ConvertTensor(y), ConvertTensor(z));
        aclDumpOpTensors("Add", "add_custom", tensors, 2, 1, aclStream);

        // Destroy the aclTensor object.
        for (size_t i = 0; i < 3; i++) {
            aclDestroyTensor(tensors[i]);
        }

        return z;
    }
    } // namespace ascendc_ops

    TORCH_LIBRARY(ascendc_ops, m)
    {
        m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
    {
        m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
    }
    ```
