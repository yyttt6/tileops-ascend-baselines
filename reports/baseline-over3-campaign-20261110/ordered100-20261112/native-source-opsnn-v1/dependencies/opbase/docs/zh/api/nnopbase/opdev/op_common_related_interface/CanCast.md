# CanCast

**须知：该接口后续版本会废弃，请使用最新[CanCast](../data_type_utils/CanCast.md)接口**

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
// 判断dtype不可以转为int8类型时，返回
void Func(const DataType dtype) {
    if (!CanCast(dtype, DT_INT8)) {
        return;
    }
}
```
