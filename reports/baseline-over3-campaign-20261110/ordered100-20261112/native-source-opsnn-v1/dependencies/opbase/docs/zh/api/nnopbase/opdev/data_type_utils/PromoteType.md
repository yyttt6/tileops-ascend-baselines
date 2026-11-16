# PromoteType

## 功能说明

不同数据类型参数进行运算时，推导应该将类型提升到何种数据类型进行计算。例如float16与float32进行运算，应该将float16先提升至float32后再做运算。

## 函数原型

```cpp
ge::DataType PromoteType(ge::DataType type_a, ge::DataType type_b)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type_a | 输入 | 第一个参数数据类型。 |
| type_b | 输入 | 第二个参数数据类型。 |

## 返回值说明

若无法推导类型提升后的数据类型，运算会失败并返回DT\_UNDEFINED。

## 约束说明

无

## 调用示例

```cpp
// 校验dtype是否可以与Float32计算，不能则提前返回
void Func(const ge::DataType dtype) {
    if (PromoteType(dtype, DT_FLOAT) == DT_UNDEFINED) {
        return;
    }
    // 后续执行算子计算逻辑
}
```
