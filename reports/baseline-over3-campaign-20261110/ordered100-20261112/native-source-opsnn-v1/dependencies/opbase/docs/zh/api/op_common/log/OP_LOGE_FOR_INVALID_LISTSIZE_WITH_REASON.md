# OP\_LOGE\_FOR\_INVALID\_LISTSIZE\_WITH\_REASON

## 功能说明

记录并上报参数列表大小校验错误（带原因）。当算子的指定参数列表大小无效时，输出ERROR级别日志并上报EZ0038错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(entityName, paramName, incorrectSize, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectSize | 输入 | 实际列表大小，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (listSize > maxListSize) {
    OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON("MyOp", "dims", std::to_string(listSize).c_str(), "exceeds max limit");
    return ge::GRAPH_FAILED;
}
```
