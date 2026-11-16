# IsFloatEqual

## Function

Checks if two `float` or `double` values are equal.

## Prototype

```cpp
template <typename T>
auto IsFloatEqual(T a, T b) -> typename std::enable_if<std::is_floating_point<T>::value, bool>::type
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| a | Input| Value to be compared. The data type can be `float` or `double`.|
| b | Input| Value to be compared. The data type can be `float` or `double`.|

## Returns

`bool` value. **true** indicates that the two `float` or `double` values are equal, and **false** indicates that they are not equal.

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

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
