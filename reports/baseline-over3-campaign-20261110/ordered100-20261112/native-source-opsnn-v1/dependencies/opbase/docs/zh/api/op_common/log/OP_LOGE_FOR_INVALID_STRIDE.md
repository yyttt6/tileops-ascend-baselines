# OP\_LOGE\_FOR\_INVALID\_STRIDE

## 功能说明

记录并上报参数stride校验错误。当算子的指定参数stride与预期不符时，输出ERROR级别日志并上报EZ0028错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_STRIDE(entityName, paramName, incorrectStride, correctStride)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectStride | 输入 | 实际stride，格式如"[1,1,1,1]"，支持const char*或std::string类型。 |
| correctStride | 输入 | 预期stride，格式如"[1,1,1,1]"，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (!CheckStrideValid(inputStride, expectedStride)) {
    OP_LOGE_FOR_INVALID_STRIDE("MyOp", "x", StrideToStr(inputStride).c_str(), "[1,1,1,1]");
    return ge::GRAPH_FAILED;
}
```
