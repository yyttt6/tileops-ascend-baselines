# IsFloatingType

## 功能说明

判断输入的数据类型是否为浮点类型，包括Float64（即Double）、Float32（即Float）、BFloat16、Float16，以及HiFloat8、Float8E5M2、Float8E4M3FN、Float8E8M0、Float6E3M2、Float6E2M3、Float4E2M1、Float4E1M2。

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
// 校验dtype是否为浮点数类型，不是则提前返回
void Func(const ge::DataType dtype) {
    if (!IsFloatingType(dtype)) {
        return;
    }
    // 后续执行算子计算逻辑
}
```
