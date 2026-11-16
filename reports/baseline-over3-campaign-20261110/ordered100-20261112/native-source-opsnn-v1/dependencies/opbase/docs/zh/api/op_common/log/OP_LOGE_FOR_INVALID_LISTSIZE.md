# OP\_LOGE\_FOR\_INVALID\_LISTSIZE

## 功能说明

记录并上报参数列表大小校验错误。当算子的指定参数列表大小（元素个数）与预期不符时，输出ERROR级别日志并上报EZ0025错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_LISTSIZE(entityName, paramName, incorrectSize, correctSize)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectSize | 输入 | 实际列表大小，支持const char*或std::string类型。 |
| correctSize | 输入 | 预期列表大小，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter axes of SoftmaxV2 has incorrect element nums 2. It should be 1.
if (axes->GetSize() != 1) {
    OP_LOGE_FOR_INVALID_LISTSIZE("SoftmaxV2", "axes",
        std::to_string(axes->GetSize()), "1");
    return ge::GRAPH_FAILED;
}
```
