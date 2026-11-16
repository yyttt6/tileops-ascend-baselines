# SetUnknownRank

## 功能说明

图模式场景下，当输入张量shape为未知秩，需通过本接口将输出张量shape也设为未知秩。

## 函数原型

```cpp
void SetUnknownRank(gert::Shape &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输出 | 输出张量的shape。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto in_shape = context->GetInputShape(0);  // 0表示第一个输入参数
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0);  // 0表示第一个输出参数
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

// 判断输入张量shape是否为未知秩，若是，将输出张量shape置为未知秩
if (Ops::Base::IsUnknownRank(*in_shape)) {
    Ops::Base::SetUnknownRank(*out_shape);
}
```
