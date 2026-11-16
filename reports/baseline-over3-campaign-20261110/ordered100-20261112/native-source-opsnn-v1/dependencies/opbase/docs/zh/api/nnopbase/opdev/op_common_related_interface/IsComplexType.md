# IsComplexType

**须知：该接口后续版本会废弃，请使用最新[IsComplexType](../data_type_utils/IsComplexType.md)接口**

## 功能说明

判断输入的数据类型是否为复数类，包括Complex128、Complex64、Complex32。

## 函数原型

```cpp
bool IsComplexType(const ge::DataType type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type | 输入 | 输入的数据类型。 |

## 返回值说明

若输入的数据类型为复数类型返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断dtype不为复数类型时，返回
void Func(const DataType dtype) {
    if (!IsComplexType(dtype)) {
        return;
    }
}
```
