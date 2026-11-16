# IsUnknownShape

## 功能说明

图模式场景下，检查输入shape的每一根轴长度是否都为不确定值。

## 函数原型

```cpp
bool IsUnknownShape(const gert::Shape &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 待判定的输入shape。 |

## 返回值说明

返回类型为bool。

- true：输入shape的每一根轴长度都为不确定值。
- false：输入shape至少有1根轴的长度为确定值。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
if (Ops::Base::IsUnknownShape(*inputShape)) {
    output_y_shape->SetDimNum(input_dim_num);
}
```
