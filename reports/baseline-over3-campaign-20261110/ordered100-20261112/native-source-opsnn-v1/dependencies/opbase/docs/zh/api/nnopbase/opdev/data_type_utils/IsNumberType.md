# IsNumberType

## 功能说明

判断输入的数据类型，是否为数字类型。

## 函数原型

```cpp
bool IsNumberType(const DataType dtype)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dtype | 输入 | 输入数据类型，判断该类型是否为数字类型。<br>数字类型包括：Complex128、Complex64、Float64、Float32、Float16、Float8_E5M2、Float8_E4M3FN、Float8_E8M0、Float6_E3M2、Float6_E2M3、Float4_E2M1、Float4_E1M2、HiFloat8、Int16、Int32、Int64、Int8、QInt32、QInt8、QUInt8、UInt16、UInt32、UInt64、UInt8、BFloat16。 |

## 返回值说明

输入数据类型为数字类型时，返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断dtype不为数字类型时，返回
void Func(const DataType dtype) {
    if (!IsNumberType(dtype)) {
        return;
    }
}
```
