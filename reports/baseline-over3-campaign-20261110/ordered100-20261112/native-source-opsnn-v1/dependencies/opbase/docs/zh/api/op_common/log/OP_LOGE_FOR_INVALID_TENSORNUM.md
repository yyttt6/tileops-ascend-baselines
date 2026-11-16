# OP\_LOGE\_FOR\_INVALID\_TENSORNUM

## 功能说明

记录并上报参数张量数量校验错误。当算子的指定参数张量数量与预期不符时，输出ERROR级别日志并上报EZ0022错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_TENSORNUM(entityName, paramName, incorrectNum, correctNum)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectNum | 输入 | 实际张量数量，int64_t类型。 |
| correctNum | 输入 | 预期张量数量，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter instance of Foreach has invalid tensor num 1000.
//          It should be within the range [1, 950].
if (tensorCount > MAX_COUNT || tensorCount <= 0) {
    OP_LOGE_FOR_INVALID_TENSORNUM("Foreach", "instance",
        static_cast<int64_t>(tensorCount),
        ("within the range [1, " + std::to_string(MAX_COUNT) + "]").c_str());
    return ge::GRAPH_FAILED;
}
```
