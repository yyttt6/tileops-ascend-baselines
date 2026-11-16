# OP\_LOGE\_FOR\_FILE\_PARSE

## 功能说明

记录并上报文件解析失败错误。当文件内容解析失败时，输出ERROR级别日志并上报EZ0031错误码。

## 函数原型

```cpp
OP_LOGE_FOR_FILE_PARSE(opName, fileName, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| fileName | 输入 | 文件名称，支持const char*或std::string类型。 |
| reason | 输入 | 解析失败原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (!ParseJsonFile(fileName, config)) {
    OP_LOGE_FOR_FILE_PARSE("MyOp", fileName, "invalid json format");
    return ge::GRAPH_FAILED;
}
```
