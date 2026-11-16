# GetOriginalFormat

## 功能说明

获取aclTensor的OriginFormat。

OriginFormat一般表示aclTensor在经历transdata节点前（如果存在该节点）的原始Format信息。

## 函数原型

```cpp
op::Format GetOriginalFormat()
```

## 参数说明

无

## 返回值说明

返回一个op::Format（即ge::Format），本身是一个枚举，包含多种不同的Format，例如NCHW、ND等。

> **说明**：ge::Format介绍参见[《基础数据结构和接口参考》](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi)中”ge命名空间 > Format“。

## 约束说明

无

## 调用示例

```cpp
void Func(const aclTensor *input) {
    auto format = input->GetOriginalFormat();
}
```
