# IsIntegralType（含bool）

**须知：该接口后续版本会废弃，请使用最新[IsIntegralType（含bool）](../data_type_utils/IsIntegralType_bool.md)接口**

## 功能说明

判断输入的数据类型是否为整数类型，包括Int8、Int16、Int32、Int64、Uint8、Uint16、Uint32、Uint64。

若include\_bool置为true，那么bool也被认为是整数。

## 函数原型

```cpp
bool IsIntegralType(const ge::DataType type, const bool include_bool)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type | 输入 | 输入的数据类型。 |
| include_bool | 输入 | 是否将bool视为整数类型。 |

## 返回值说明

若为整数类型返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断dtype不为整数类型时，返回，注意bool也是整数类型
void Func(const ge::DataType type) {
    if (!IsIntegralType(type, true)) {
        return;
    }
}
```
