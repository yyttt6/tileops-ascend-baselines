# SetStorageFormat

## 功能说明

设置aclTensor的StorageFormat。

StorageFormat表示aclTensor在内存中的排布格式，例如NCHW、ND等。

## 函数原型

```cpp
void SetStorageFormat(op::Format format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 数据类型为op::Format（即ge::Format），本身是一个枚举，包含多种不同的Format，例如NCHW、ND等。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 将input的Storage Format置为ND格式
void Func(const aclTensor *input) {
    input->SetStorageFormat(ge::FORMAT_ND);
}
```
