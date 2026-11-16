# SetUnknownShape

## 功能说明

图模式场景下，设置输入shape的维度为rank，且每根轴长度都为不确定值。

## 函数原型

```cpp
void SetUnknownShape(int64_t rank, gert::Shape &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| rank | 输入 | 设定shape的维度。 |
| shape | 输出 | shape的维度为rank，且每根轴长度都为不确定值。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
auto in_shape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
auto out_shape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

if (Ops::Base::IsUnknownShape(*in_shape)) {
    Ops::Base::SetUnknownShape(0, *out_shape);
}
```
