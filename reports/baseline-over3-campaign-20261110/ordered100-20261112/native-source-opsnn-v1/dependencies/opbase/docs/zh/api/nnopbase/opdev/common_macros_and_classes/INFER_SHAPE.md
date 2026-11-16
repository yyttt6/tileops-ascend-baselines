# INFER\_SHAPE

## 宏功能

针对指定算子，运行其InferShape函数，推导输出shape。

## 宏原型

```cpp
INFER_SHAPE(KERNEL_NAME, op_args...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| KERNEL_NAME | 输入 | 算子名，例如Add。 |
| op_args... | 输入 | 算子的参数，包括输入[OP_INPUT](OP_INPUT.md)、输出[OP_OUTPUT](OP_OUTPUT.md)、属性[OP_ATTR](OP_ATTR.md)等参数。 |

## 约束说明

- 如果算子需要INFER\_SHAPE，那么此宏需要在[ADD\_TO\_LAUNCHER\_LIST\_AICORE](ADD_TO_LAUNCHER_LIST_AICORE.md)之前调用。
- 如下接口是上述宏定义会调用到的关联接口。

    ```text
    OP_INPUT(x...)
    OP_OUTPUT(x...)
    OP_ATTR(x...)
    OP_WORKSPACE(x...)
    OP_OUTSHAPE(x...)
    OP_OPTION(x...)
    OP_EMPTY_ARG
    OP_MODE(x...)
    ```

## 调用示例

```cpp
// 调用INFER_SHAPE推导batchmatmul算子的输出shape，其中BatchMatMulV3是算子的名字，OP_INPUT是算子输入参数，OP_OUTPUT是算子输出参数，OP_ATTR是算子的属性参数
INFER_SHAPE(BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum));
```
