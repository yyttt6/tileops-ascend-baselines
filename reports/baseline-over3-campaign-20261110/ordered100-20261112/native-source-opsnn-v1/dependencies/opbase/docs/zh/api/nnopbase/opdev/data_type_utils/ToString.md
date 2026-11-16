# ToString

## 功能说明

将一组数据类型序列化为字符串形式。空列表返回"[]"；非空列表返回形如"[DT_FLOAT,DT_INT8,]"的字符串，列表中每个数据类型以英文逗号分隔。

## 函数原型

```cpp
ge::AscendString ToString(const std::initializer_list<DataType>& dataTypes)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dataTypes | 输入 | 待序列化的数据类型列表。 |

## 返回值说明

返回序列化后的字符串，类型为`ge::AscendString`。空列表返回"[]"；非空列表返回"[DT_FLOAT,DT_INT8,]"形式的字符串，每个元素后均带英文逗号。

## 约束说明

无

## 调用示例

```cpp
// 将一组数据类型序列化为字符串并打印
void Func() {
    ge::AscendString typeStr = ToString({DT_FLOAT, DT_INT8});
    // typeStr内容为"[DT_FLOAT,DT_INT8,]"
}
```
