# ADD\_TO\_LAUNCHER\_LIST\_AICORE

## 宏功能

创建某个AI Core算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段接口aclnn_Xxx_调用时执行。

## 宏原型

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| KERNEL_NAME | 输入 | 算子名，例如Add。 |
| op_args... | 输入 | 算子的参数，包括输入[OP_INPUT](OP_INPUT.md)、输出[OP_OUTPUT](OP_OUTPUT.md)、属性[OP_ATTR](OP_ATTR.md)等参数。 |

## 约束说明

- 如果算子需要INFER\_SHAPE，那么此宏需要在[INFER\_SHAPE](INFER_SHAPE.md)之后调用。
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
// 调用ADD_TO_LAUNCHER_LIST_AICORE创建add算子的执行任务，其中Add是算子名，self和other是算子输入参数，addOut是算子输出参数
ADD_TO_LAUNCHER_LIST_AICORE(Add, OP_INPUT(self, other), OP_OUTPUT(addOut));
```
