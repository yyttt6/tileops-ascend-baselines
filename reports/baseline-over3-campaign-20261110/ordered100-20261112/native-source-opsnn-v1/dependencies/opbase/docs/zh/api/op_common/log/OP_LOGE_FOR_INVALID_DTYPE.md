# OP\_LOGE\_FOR\_INVALID\_DTYPE

## 功能说明

记录并上报参数数据类型校验错误。当算子的指定参数数据类型与预期不符时，输出ERROR级别日志并上报EZ0019错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_DTYPE(entityName, paramName, incorrectDtype, correctDtype)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectDtype | 输入 | 实际数据类型（如"INT32"），支持const char*或std::string类型。 |
| correctDtype | 输入 | 预期数据类型（如"FLOAT16"），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter softmax of SoftmaxGrad has incorrect dtype INT32. It should be
//          FLOAT, FLOAT16 or BF16.
if (dtype_ != ge::DT_FLOAT && dtype_ != ge::DT_FLOAT16 && dtype_ != ge::DT_BF16) {
    OP_LOGE_FOR_INVALID_DTYPE("SoftmaxGrad", "softmax",
        Ops::Base::ToString(dtype_).c_str(), "FLOAT, FLOAT16 or BF16");
    return ge::GRAPH_FAILED;
}
```
