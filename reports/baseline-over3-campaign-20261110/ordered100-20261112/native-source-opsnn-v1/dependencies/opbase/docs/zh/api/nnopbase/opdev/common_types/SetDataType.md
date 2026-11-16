# SetDataType

## 功能说明

设置aclTensor的数据类型。

## 函数原型

```cpp
void SetDataType(op::DataType dataType)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dataType | 输入 | 数据类型为op::DataType（即ge::DataType），将aclTensor设置为dataType类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 将input的数据类型设置为int64
void Func(const aclTensor *input) {
    input->SetDataType(DT_INT64);
}
```
