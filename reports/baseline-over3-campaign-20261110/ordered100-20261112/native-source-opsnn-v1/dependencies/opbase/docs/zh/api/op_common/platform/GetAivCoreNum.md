# GetAivCoreNum

## 功能说明

获取平台AI Vector的核数。

## 函数原型

```cpp
template <typename T>
uint32_t GetAivCoreNum(const T *context)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输入 | Tiling的上下文信息，类型为TilingContext/TilingParseContext。 |

## 返回值说明

返回平台AI Vector的核数，uint32\_t类型，返回0代表获取失败。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
compileInfo->vectorCoreNum = GetAivCoreNum(context_);
```
