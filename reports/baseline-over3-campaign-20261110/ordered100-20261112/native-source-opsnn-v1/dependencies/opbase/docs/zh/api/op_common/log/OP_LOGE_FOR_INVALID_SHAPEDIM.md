# OP\_LOGE\_FOR\_INVALID\_SHAPEDIM

## 功能说明

记录并上报参数形状维度校验错误。当算子的指定参数形状维度与预期不符时，输出ERROR级别日志并上报EZ0011错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_SHAPEDIM(entityName, paramName, incorrectDim, correctDim)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectDim | 输入 | 实际维度（如"3D"），支持const char*或std::string类型。 |
| correctDim | 输入 | 预期维度（如"4D"），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter x of SoftmaxV2 has incorrect shape dim 5D. It should be less than
//          or equal to 4.
if (xShape.GetDimNum() > 4) {
    std::string dimStr = std::to_string(xShape.GetDimNum()) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM("SoftmaxV2", "x", dimStr.c_str(),
        "less than or equal to 4");
    return ge::GRAPH_FAILED;
}
```
