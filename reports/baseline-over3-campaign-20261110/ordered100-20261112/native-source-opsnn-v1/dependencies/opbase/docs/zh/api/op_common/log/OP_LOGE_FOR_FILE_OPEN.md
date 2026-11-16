# OP\_LOGE\_FOR\_FILE\_OPEN

## 功能说明

记录并上报文件打开失败错误。当文件无法打开时，输出ERROR级别日志并上报EZ0030错误码。

## 函数原型

```cpp
OP_LOGE_FOR_FILE_OPEN(opName, filePath, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| filePath | 输入 | 文件路径，支持const char*或std::string类型。 |
| reason | 输入 | 打开失败原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
std::ifstream ifs(filePath);
if (!ifs.is_open()) {
    OP_LOGE_FOR_FILE_OPEN("MyOp", filePath, "permission denied");
    return ge::GRAPH_FAILED;
}
```
