# GetViewFormat

## 功能说明

获取aclTensor的ViewFormat。ViewFormat为aclTensor的逻辑Format。

## 函数原型

```cpp
op::Format GetViewFormat()
```

## 参数说明

无

## 返回值说明

返回一个op::Format（即ge::Format），本身是一个枚举，包含多种不同的Format，例如NCHW、ND等。

> **说明**： ge::Format介绍参见[《基础数据结构和接口参考》](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi)中”ge命名空间 > Format“。

## 约束说明

无

## 调用示例

```cpp
void Func(const aclTensor *input) {
    auto format = input->GetViewFormat();
}
```
