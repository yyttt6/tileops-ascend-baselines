# OP\_LOGE\_WITH\_INVALID\_INPUT\_DTYPE

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_DTYPE](OP_LOGE_FOR_INVALID_DTYPE.md) 替代。

## 功能说明

记录并上报算子输入数据类型校验错误。当算子的指定输入参数数据类型与预期不符时，输出ERROR级别日志并上报EZ0007错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_INPUT_DTYPE(entityName, paramName, dataDtype, expectedDtypeList)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 输入参数名称，支持const char*或std::string类型。 |
| dataDtype | 输入 | 实际数据类型（如"INT32"），支持const char*或std::string类型。 |
| expectedDtypeList | 输入 | 预期数据类型列表（如"FLOAT16, FLOAT"），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Input parameter x of SoftmaxV2 has incorrect dtype INT32. It should be FLOAT, FLOAT16 or BF16.
if (dtype_ != FLOAT && dtype_ != FLOAT16 && dtype_ != BF16) {
    OP_LOGE_WITH_INVALID_INPUT_DTYPE("SoftmaxV2", "x",
        Ops::Base::ToString(dtype_).c_str(), "FLOAT, FLOAT16 or BF16");
    return ge::GRAPH_FAILED;
}
```
