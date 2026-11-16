# IsFloatEqual

## 功能说明

判断两个float类型或double类型的数值是否相等。

## 函数原型

```cpp
template <typename T>
auto IsFloatEqual(T a, T b) -> typename std::enable_if<std::is_floating_point<T>::value, bool>::type
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| a | 输入 | 待比较参数，数据类型支持float、double。 |
| b | 输入 | 待比较参数，数据类型支持float、double。 |

## 返回值说明

返回bool类型。true表示两个float或double类型数值相等；flase表示两个float或double类型数值不相等。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
float delta;
if (op.GetAttr("delta", delta) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("delta");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
}
if (IsFloatEqual(delta, 0.0f)) {
    string excepted_value = ConcatString("not equal to 0");
    std::string err_msg = GetAttrValueErrMsg("delta", ConcatString("delta"), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
}
```
