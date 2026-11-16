# OP\_OUTSHAPE

## Function

Stores the aclTensor of the output shape of an operator, for example, the NonZero operator.

## Prototype

```cpp
OP_OUTSHAPE(x...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| x... | Input| The parameter consists of two parts. The first parameter is the aclTensor that stores the output tensor shape, and the second parameter is the index of the tensor whose output shape needs to be updated.|

## Restrictions

Only one output tensor shape (outShapeTensor) can be updated with a shape of (9\*Number of tensors to be updated,). The shape of each output tensor occupies nine bits. The first bit indicates the number of dimensions, and the remaining eight bits indicate the value of each dimension.

## Examples

```cpp
// The operator stores the shape of the output tensor in outShapeTensor to update the shape of the output tensor whose index is 0.
OP_OUTSHAPE({outShapeTensor, 0});

// Refer to the following example to update the shapes of multiple output tensors. In this example, the shapes of the output tensors whose indexes are 0, 3, and 4 need to be updated.
// On the host:
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

// On the kernel:
__aicore__ inline void CopyOutShape(uint64_t dimNums1, uint64_t *dimNums2, uint64_t dimNums3)
{
    LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
    shapeTensor.SetValue(0, 1);                // Dimension information of the first output tensor
    shapeTensor.SetValue(1, dimNums1);         // Shape value of the first dimension of the first output tensor
    shapeTensor.SetValue(9, 1);               // Dimension information of the second output tensor
    shapeTensor.SetValue(10, *(dimNums2));     // Shape value of the first dimension of the second output tensor
    shapeTensor.SetValue(11, *(dimNums2+1));   // Shape value of the second dimension of the second output tensor
    shapeTensor.SetValue(18, 1);               // Dimension information of the third output tensor
    shapeTensor.SetValue(19, dimNums3);        // Shape value of the first dimension of the third output tensor
    ...
    DataCopyPad(...);
}
```
