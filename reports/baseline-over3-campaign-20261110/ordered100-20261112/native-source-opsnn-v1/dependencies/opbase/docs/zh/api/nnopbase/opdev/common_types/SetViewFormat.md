# SetViewFormat

## 功能说明

设置aclTensor的ViewFormat。ViewFormat为aclTensor的逻辑Format。

## 函数原型

```cpp
void SetViewFormat(op::Format format)
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
// 将input的ViewFormat置为ND格式
void Func(const aclTensor *input) {
    input->SetViewFormat(ge::FORMAT_ND);
}
```
