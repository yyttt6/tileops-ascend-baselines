# IsComplexType

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
// 校验dtype是否为复数类型，不是则提前返回
void Func(const ge::DataType dtype) {
    if (!IsComplexType(dtype)) {
        return;
    }
    // 后续执行算子计算逻辑
}
```
