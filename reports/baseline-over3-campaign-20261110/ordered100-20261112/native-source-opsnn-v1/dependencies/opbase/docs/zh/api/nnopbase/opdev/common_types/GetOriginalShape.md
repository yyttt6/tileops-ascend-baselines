# GetOriginalShape

## 功能说明

获取aclTensor的OriginShape属性。

OriginShape表示aclTensor在经历transdata节点前（如果存在该节点）的原始shape信息，即tensor的shape数学描述。

## 函数原型

```cpp
gert::Shape GetOriginalShape()
```

## 参数说明

无

## 返回值说明

返回值为gert::Shape，记录了一组shape信息，例如一个三维shape：\[10, 20, 30\]。

> **说明**：gert::Shape介绍参见[《基础数据结构和接口参考》](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi)中”gert命名空间 > Shape“。

## 约束说明

无

## 调用示例

```cpp
void Func(const aclTensor *input) {
    auto shape = input->GetOriginalShape();
}
```
