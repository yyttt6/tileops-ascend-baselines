# OP\_LOGE\_FOR\_FILE\_PATH

## 功能说明

记录并上报文件路径无效错误。当输入文件路径无效时，输出ERROR级别日志并上报EZ0029错误码。

## 函数原型

```cpp
OP_LOGE_FOR_FILE_PATH(opName, filePath, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| filePath | 输入 | 文件路径，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (access(filePath, F_OK) != 0) {
    OP_LOGE_FOR_FILE_PATH("MyOp", filePath, "file does not exist");
    return ge::GRAPH_FAILED;
}
```
