# TypeSize

## 功能说明

计算输入的数据类型一个元素占用多少Byte。对于不满一个Byte大小的数据类型，会在基础bit大小上额外加1000，例如Int4，将返回1004，表示一个元素占用4bit。

## 函数原型

```cpp
size_t TypeSize(DataType dataType)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dataType | 输入 | 输入的数据类型，用于计算该数据类型单元素的大小。 |

## 返回值说明

返回输入数据类型单元素的大小。

## 约束说明

无

## 调用示例

```cpp
// 获取dtype的单个元素size大小
void Func(const DataType dtype) {
    size_t size = TypeSize(dtype);
}
```
