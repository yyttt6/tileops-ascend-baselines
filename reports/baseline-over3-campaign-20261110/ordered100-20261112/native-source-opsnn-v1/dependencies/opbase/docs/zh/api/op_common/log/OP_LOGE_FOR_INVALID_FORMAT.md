# OP\_LOGE\_FOR\_INVALID\_FORMAT

## 功能说明

记录并上报参数格式校验错误。当算子的指定参数格式与预期不符时，输出ERROR级别日志并上报EZ0017错误码。

## 函数原型

```cpp
OP_LOGE_FOR_INVALID_FORMAT(entityName, paramName, incorrectFormat, correctFormat)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 参数名称，支持const char*或std::string类型。 |
| incorrectFormat | 输入 | 实际格式（如"NHWC"），支持const char*或std::string类型。 |
| correctFormat | 输入 | 预期格式（如"NCHW"），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Parameter x of ResizeBilinearV2 has incorrect format ND. It should be
//          NCHW or NHWC.
if (format_ != ge::FORMAT_NCHW && format_ != ge::FORMAT_NHWC) {
    OP_LOGE_FOR_INVALID_FORMAT("ResizeBilinearV2", "x",
        Ops::Base::ToString(format_).c_str(), "NCHW or NHWC");
    return ge::GRAPH_FAILED;
}
```
