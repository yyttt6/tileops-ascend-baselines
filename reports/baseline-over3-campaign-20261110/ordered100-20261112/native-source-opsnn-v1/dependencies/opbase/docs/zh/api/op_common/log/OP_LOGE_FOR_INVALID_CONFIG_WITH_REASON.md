# OP\_LOGE\_FOR\_INVALID\_CONFIG\_WITH\_REASON

## 功能说明

记录并上报配置项值无效错误（带原因）。当配置文件中某项的值无效时，输出ERROR级别日志并上报EZ0033错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, configFile, item, value, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| configFile | 输入 | 配置文件路径，支持const char*或std::string类型。 |
| item | 输入 | 配置项名称，支持const char*或std::string类型。 |
| value | 输入 | 实际值，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (!IsValidConfig(value)) {
    OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON("MyOp", "config.ini", "block_size", value.c_str(), "out of range");
    return ge::GRAPH_FAILED;
}
```
