# OP\_LOGE\_FOR\_INVALID\_TENSORNUMS\_WITH\_REASON

## 功能说明

记录并上报多个参数张量数量校验错误（附带原因说明）。当算子的多个参数张量数量不正确时，输出ERROR级别日志（含具体原因）并上报EZ0023错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(entityName, paramNames, incorrectNums, reason)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramNames | 输入 | 参数名称列表，支持const char*或std::string类型。 |
| incorrectNums | 输入 | 实际张量数量列表，支持const char*或std::string类型。 |
| reason | 输入 | 错误原因描述，支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameters lse and go of AttentionUpdate have invalid tensor nums 4 and 4.
//          Reason: The number of tensors in input lse and go should be twice the attr sp, where sp is 2.
if (lseCount != sp * 2 || goCount != sp * 2) {
    std::string numMsg = std::to_string(lseCount) + " and " + std::to_string(goCount);
    OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON("AttentionUpdate", "lse and go",
        numMsg.c_str(),
        "The number of tensors in input lse and go should be twice the attr sp, where sp is 2.");
    return ge::GRAPH_FAILED;
}
```
