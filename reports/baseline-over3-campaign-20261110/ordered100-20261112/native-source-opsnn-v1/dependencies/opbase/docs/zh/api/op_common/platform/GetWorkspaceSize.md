# GetWorkspaceSize

## 功能说明

获取平台LibApi Workspace的大小，单位bytes。

## 函数原型

```cpp
template <typename T>
uint32_t GetWorkspaceSize(const T *context)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输入 | Tiling的上下文信息，类型为TilingContext/TilingParseContext。 |

## 返回值说明

uint32\_t：返回平台LibApi Workspace大小，返回0代表获取失败。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
compileInfo->workspaceSize = GetWorkspaceSize(context_);
```
