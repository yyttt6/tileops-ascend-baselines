# GetDataType

## 功能说明

获取aclTensor的数据类型。

## 函数原型

```cpp
op::DataType GetDataType()
```

## 参数说明

无

## 返回值说明

op::DataType（即ge::DataType）表示aclTensor中元素的数据类型，包含多种数据类型，例如float16、float32等。

> **说明**：ge::DataType介绍参见[《基础数据结构和接口参考》](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi)中”ge命名空间 > DataType“。

## 约束说明

无

## 调用示例

```cpp
// 获取input的数据类型
void Func(const aclTensor *input) {
    op::DataType dataType = input->GetDataType();
}
```
