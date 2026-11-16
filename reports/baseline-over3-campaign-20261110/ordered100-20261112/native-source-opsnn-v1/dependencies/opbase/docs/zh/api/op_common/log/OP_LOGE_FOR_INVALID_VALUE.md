# OP\_LOGE\_FOR\_INVALID\_VALUE

## 功能说明

记录并上报参数值校验错误。当算子的指定参数值与预期不符时，输出ERROR级别日志并上报EZ0024错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_VALUE(entityName, paramName, incorrectValue, correctValue)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectValue | 输入 | 实际参数值，支持const char*或std::string类型。 |
| correctValue | 输入 | 预期参数值，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter sp of AttentionUpdate has incorrect value 17. It should be
//          in range of [1, 16].
if (sp_ < 1 || sp_ > 16) {
    OP_LOGE_FOR_INVALID_VALUE("AttentionUpdate", "sp", std::to_string(sp_), "in range of [1, 16]");
    return ge::GRAPH_FAILED;
}
```
