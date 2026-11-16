# OP\_LOGE\_WITH\_INVALID\_INPUT\_FORMAT

> **说明：** 本接口已废弃，建议使用 [OP_LOGE_FOR_INVALID_FORMAT](OP_LOGE_FOR_INVALID_FORMAT.md) 替代。

## 功能说明

记录并上报算子输入格式校验错误。当算子的指定输入参数格式与预期不符时，输出ERROR级别日志并上报EZ0006错误码。

## 函数原型

```cpp
OP_LOGE_WITH_INVALID_INPUT_FORMAT(entityName, paramName, dataFormat, expectedFormatList)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| entityName | 输入 | 算子名称或aclnn接口名称，支持const char*或std::string类型。 |
| paramName | 输入 | 输入参数名称，支持const char*或std::string类型。 |
| dataFormat | 输入 | 实际数据格式（如"NHWC"），支持const char*或std::string类型。 |
| expectedFormatList | 输入 | 预期格式列表（如"NCHW, NCDHW"），支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 预期输出: Input parameter x of ResizeBilinearV2 has incorrect format ND. It should be NCHW or NHWC.
if (format_ != NCHW && format_ != NHWC) {
    OP_LOGE_WITH_INVALID_INPUT_FORMAT("ResizeBilinearV2", "x",
        Ops::Base::ToString(format_).c_str(), "NCHW or NHWC");
    return ge::GRAPH_FAILED;
}
```
