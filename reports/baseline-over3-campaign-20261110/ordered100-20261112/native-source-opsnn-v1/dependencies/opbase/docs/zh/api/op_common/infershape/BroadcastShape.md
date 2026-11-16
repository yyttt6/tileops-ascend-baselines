# BroadcastShape

## 功能说明

根据输入张量的shape推导broadcast后的输出shape。支持双输入和多输入两种重载。

## 函数原型

```cpp
bool BroadcastShape(const gert::Shape* in1Shape, const gert::Shape* in2Shape, gert::Shape* outShape)
```

```cpp
bool BroadcastShape(const std::vector<const gert::Shape*>& inShapes, gert::Shape* outShape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| in1Shape | 输入 | 第一个输入张量的shape。 |
| in2Shape | 输入 | 第二个输入张量的shape。 |
| inShapes | 输入 | 多输入场景下，所有输入张量shape的列表。 |
| outShape | 输出 | 推导后的输出shape。 |

## 返回值说明

bool类型：

- true：broadcast推导成功。
- false：broadcast推导失败，输入shape无法进行broadcast。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 双输入场景
auto in1Shape = context->GetInputShape(0);
auto in2Shape = context->GetInputShape(1);
auto outShape = context->GetOutputShape(0);
if (!Ops::Base::BroadcastShape(in1Shape, in2Shape, outShape)) {
    return ge::GRAPH_FAILED;
}

// 多输入场景
std::vector<const gert::Shape*> inShapes = {context->GetInputShape(0), context->GetInputShape(1), context->GetInputShape(2)};
auto outShape = context->GetOutputShape(0);
if (!Ops::Base::BroadcastShape(inShapes, outShape)) {
    return ge::GRAPH_FAILED;
}
```
