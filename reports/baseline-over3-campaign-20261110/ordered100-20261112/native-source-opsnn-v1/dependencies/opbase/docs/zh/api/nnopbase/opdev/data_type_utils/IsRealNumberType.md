# IsRealNumberType

## 功能说明

判断输入的数据类型，是否只包含实部数字类型。

## 函数原型

```cpp
bool IsRealNumberType(const DataType dtype)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dtype | 输入 | 输入数据类型，判断该类型是否仅包含实部。<br>实部数字类型包括：Float64、Float32、Float16、Float8_E5M2、Float8_E4M3FN、Float8_E8M0、Float6_E3M2、Float6_E2M3、Float4_E2M1、Float4_E1M2、HiFloat8、Int16、Int32、Int64、Int8、UInt16、UInt32、UInt64、UInt8、BFloat16。 |

## 返回值说明

输入数据类型只包含实部时，返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断dtype不只包含实部时，返回
void Func(const DataType dtype) {
    if (!IsRealNumberType(dtype)) {
        return;
    }
}
```
