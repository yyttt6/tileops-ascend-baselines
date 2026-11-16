# OP\_OUTSHAPE

## 宏功能

针对需要计算结果来确定输出shape的算子，如NonZero算子，该宏用于存放此类算子输出shape的aclTensor。

## 宏原型

```cpp
OP_OUTSHAPE(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 包含两部分，第一个参数是存放输出tensor shape的aclTensor，第二个参数是存放需要更新输出shape的tensor索引。 |

## 约束说明

只支持存放一个输出tensor shape的Tensor(outShapeTensor), 其对应shape为(9\*需要刷新的tensor个数,)。每个输出tensor的shape占9位，其中第一位表示维数，剩下8位表示具体的每个维度的具体值。

## 调用示例

```cpp
// 表示算子将输出tensor的shape存放到outShapeTensor中，并且用来更新idx=0的输出tensor的shape
OP_OUTSHAPE({outShapeTensor, 0});

// 刷新多个输出tensor的shape的时候，可以参考如下示例,本例中需要更新idx=0,3,4三个输出tensor的shape
// host侧
Shape outShapeShape{27};
auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
aclnnStatus ret =
    ADD_TO_LAUNCHER_LIST_AICORE(
        xxx,
        OP_INPUT(...),
        OP_OUTPUT(...),
        OP_ATTR(...),
        OP_OUTSHAPE({outShapeTensor, 0}),
        OP_OUTSHAPE({outShapeTensor, 3}),
        OP_OUTSHAPE({outShapeTensor, 4}),
        );

// kernel侧
__aicore__ inline void CopyOutShape(uint64_t dimNums1, uint64_t *dimNums2, uint64_t dimNums3)
{
    LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
    shapeTensor.SetValue(0, 1);                // 第一个输出tensor的维度信息
    shapeTensor.SetValue(1, dimNums1);         // 第一个输出tensor的第一维的shape值
    shapeTensor.SetValue(9, 1);                // 第二个输出tensor的维度信息
    shapeTensor.SetValue(10, *(dimNums2));     // 第二个输出tensor的第一维的shape值
    shapeTensor.SetValue(11, *(dimNums2+1));   // 第二个输出tensor的第二维的shape值
    shapeTensor.SetValue(18, 1);               // 第三个输出tensor的维度信息
    shapeTensor.SetValue(19, dimNums3);        // 第三个输出tensor的第一维的shape值
    ...
    DataCopyPad(...);
}
```
