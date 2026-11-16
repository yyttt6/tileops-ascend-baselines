# BroadcastInferShape

## 功能说明

对于满足broadcast关系（广播规则介绍参考[NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html)官网）的两个shape，推导他们broadcast后的shape。例如\[1, 10\]与\[2, 1\] broadcast后的shape为\[2, 10\]。

## 函数原型

```cpp
bool BroadcastInferShape(const op::Shape &self, const op::Shape &other, op::Shape &broadcastShape)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| self | 输入 | 第一组shape。 |
| other | 输入 | 第二组shape。 |
| broadcastShape | 输出 | self和other经过broadcast后推导的shape。 |

## 返回值说明

当self与other满足broadcast关系时，返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 生成shape为[2， 1]和[2, 10]的两个Shape对象，获取broadcast后的shape。
void Func() {
    gert::Shape shapeA;
    shapeA.AppendDim(1);
    shapeA.AppendDim(2);
    gert::Shape shapeB;
    shapeB.AppendDim(10);
    shapeB.AppendDim(2);
    gert::Shape shapeBrc;
    bool isBrc = BroadcastInferShape(shapeA, shapeB, shapeBrc);
}
```
