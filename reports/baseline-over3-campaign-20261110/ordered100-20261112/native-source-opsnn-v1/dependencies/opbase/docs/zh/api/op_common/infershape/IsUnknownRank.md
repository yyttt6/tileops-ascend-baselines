# IsUnknownRank

## 功能说明

图模式场景下，检查输入shape是否为未知秩。

## 函数原型

```cpp
bool IsUnknownRank(const gert::Shape &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 待检查的输入shape。 |

## 返回值说明

返回类型为bool。

- true：输入shape为未知秩。
- false：输入shape不为未知秩。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto in_shape = context->GetInputShape(0);  // 0表示第一个输入参数
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
// 判断输入张量shape是否为未知秩，若是，将输出张量shape置为未知秩
if (Ops::Base::IsUnknownRank(*in_shape)) {
    Ops::Base::SetUnknownRank(*out_shape);
}
```
