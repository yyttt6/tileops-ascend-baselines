# CanCast

## 功能说明

判断源数据类型是否可以cast到目的数据类型。

## 函数原型

```cpp
bool CanCast(const ge::DataType from, const ge::DataType to)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| from | 输入 | 源数据类型。 |
| to | 输入 | 目的数据类型。 |

## 返回值说明

若成功cast返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 校验dtype是否可以转为int8类型，不可以则提前返回
void Func(const ge::DataType dtype) {
    if (!CanCast(dtype, DT_INT8)) {
        return;
    }
    // 后续执行算子计算逻辑
}
```
