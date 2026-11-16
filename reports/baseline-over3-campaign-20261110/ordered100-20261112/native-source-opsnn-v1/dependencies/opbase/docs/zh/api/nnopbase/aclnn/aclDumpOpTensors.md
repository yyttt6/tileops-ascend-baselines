# aclDumpOpTensors

## 功能说明

模型执行过程中支持Dump算子输入/输出Tensor数据，方便算子输入/输出异常数据的问题定位和分析。

## 函数原型

```cpp
aclnnStatus aclDumpOpTensors(const char *opType, const char *opName, aclTensor **tensors, size_t inputTensorNum, size_t outputTensorNum, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opType | 输入 | 字符串，表示算子类型，例如“Add”。 |
| opName | 输入 | 字符串，表示算子名称，例如“add_custom”。 |
| tensors | 输入 | 一维张量，表示待Dump的输入/输出Tensor对象指针。注意Tensor顺序，输入Tensor在前，输出Tensor在后。 |
| inputTensorNum | 输入 | 表示待Dump的输入Tensor个数。 |
| outputTensorNum | 输入 | 表示待Dump的输出Tensor个数。 |
| stream | 输入 | 指定执行任务的Stream。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

本接口需要在开启算子Dump功能时有效，您可以通过aclInit接口开启Dump，也可以通过aclmdlInitDump、aclmdlSetDump、aclmdlFinalizeDump系列接口开启Dump，接口介绍请参见《Runtime运行时 API》。

## 调用示例

关键代码示例如下（仅供参考，不支持直接拷贝运行）。

1. 通过aclInit接口开启算子Dump功能。关键代码如下：

    ```cpp
    // 资源初始化
    aclInit("./acl.json");
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ```

    acl.json示例如下（具体参见aclInit接口文档中模型Dump配置、单算子Dump配置示例）：

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

2. 调用本接口关键伪代码（以torch算子为例）如下：

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

    // 生成待Dump算子的输入/输出Tensor对象指针一维张量。
    #define INIT_ACL_TENSOR_ARRAY(tensors, ...) aclTensor* tensors[] = {__VA_ARGS__}

    // at::Tensor对象转换成aclTensor对象。本函数简化了处理过程，具体以实际算子为准。
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

    // 自定义算子实现。具体以实际算子为准。
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

        // Dump算子输入/输出Tensor数据。
        INIT_ACL_TENSOR_ARRAY(tensors, ConvertTensor(x), ConvertTensor(y), ConvertTensor(z));
        aclDumpOpTensors("Add", "add_custom", tensors, 2, 1, aclStream);

        // 释放aclTensor对象。
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
