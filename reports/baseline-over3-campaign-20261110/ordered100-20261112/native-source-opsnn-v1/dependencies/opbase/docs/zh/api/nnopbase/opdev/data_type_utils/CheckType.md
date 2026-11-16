# CheckType

## 功能说明

判断给定数据类型是否在合法数据类型列表中。若在列表中找到返回true，否则返回false。

## 函数原型

```cpp
bool CheckType(const DataType dtype, const std::initializer_list<DataType>& valid_types)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dtype | 输入 | 待检查的数据类型。 |
| valid_types | 输入 | 合法数据类型列表。 |

## 返回值说明

若`dtype`在`valid_types`列表中返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 校验dtype是否为float16或float32，不是则提前返回
void Func(const ge::DataType dtype) {
    if (!CheckType(dtype, {DT_FLOAT16, DT_FLOAT})) {
        return;
    }
    // 后续执行算子计算逻辑
}
```
