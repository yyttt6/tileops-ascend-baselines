# OP\_LOGE\_FOR\_INVALID\_GRAPH\_NODE

## 功能说明

记录并上报图节点无效错误。当输入图中包含无效节点时，输出ERROR级别日志并上报EZ0036错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_GRAPH_NODE(nodeName, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| nodeName | 输入 | 图节点名称，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
if (nodeType == nullptr) {
    OP_LOGE_FOR_INVALID_GRAPH_NODE(nodeName, "node type is null");
    return ge::GRAPH_FAILED;
}
```
