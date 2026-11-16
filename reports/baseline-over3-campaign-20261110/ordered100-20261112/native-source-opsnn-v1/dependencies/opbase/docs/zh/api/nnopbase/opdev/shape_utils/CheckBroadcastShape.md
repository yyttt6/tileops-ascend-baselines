# CheckBroadcastShape

## 功能说明

校验两个shape间是否满足broadcast关系（广播规则介绍参考[NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html)官网）。例如\[2, 1\]与\[2, 10\]满足broadcast关系，\[2, 2\]与\[2, 10\]不满足broadcast关系。

## 函数原型

```cpp
bool CheckBroadcastShape(const op::Shape &self, const op::Shape &other)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| self | 输入 | 第一组shape。 |
| other | 输入 | 第二组shape。 |

## 返回值说明

当self与other满足broadcast关系时，返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 生成shape为[2， 1]和[2, 10]的两个Shape对象，校验两个shape是否满足broadcast关系。
void Func() {
    gert::Shape shapeA;
    shapeA.AppendDim(1);
    shapeA.AppendDim(2);
    gert::Shape shapeB;
    shapeB.AppendDim(10);
    shapeB.AppendDim(2);
    bool isBrc = CheckBroadcastShape(shapeA, shapeB);
}
```
