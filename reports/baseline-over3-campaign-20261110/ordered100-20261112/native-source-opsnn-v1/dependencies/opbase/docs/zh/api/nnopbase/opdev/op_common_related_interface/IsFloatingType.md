# IsFloatingType

**须知：该接口后续版本会废弃，请使用最新[IsFloatingType](../data_type_utils/IsFloatingType.md)接口**

## 功能说明

判断输入的数据类型是否为浮点类型，包括Float64（即Double）、Float32（即Float）、BFloat16、Float16。

## 函数原型

```cpp
bool IsFloatingType(const ge::DataType type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type | 输入 | 输入的数据类型。 |

## 返回值说明

若为浮点类型返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断dtype不为浮点数类型时，返回
void Func(const DataType dtype) {
    if (!IsFloatingType(dtype)) {
        return;
    }
}
```
