# GetStorageShape

## 功能说明

获取aclTensor的StorageShape属性。

StorageShape表示aclTensor在内存上的实际排布，即OriginShape实际运行时的shape格式。

## 函数原型

```cpp
gert::Shape GetStorageShape()
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
    auto shape = input->GetStorageShape();
}
```
