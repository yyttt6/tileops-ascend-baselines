# InferShape4Broadcast

## 功能说明

图模式场景下，broadcast类算子的InferShape方法（用于推导输出张量的形状shape）。支持单输入和多输入两种重载。

## 函数原型

```cpp
ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context)
```

```cpp
ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context, size_t inputNum)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输入 | GE传入的InferShape上下文。 |
| inputNum | 输入 | 参与broadcast的输入个数，用于多输入场景。 |

## 返回值说明

返回类型为ge::graphStatus：

- ge::GRAPH\_SUCCESS：InferShape成功。
- ge::GRAPH\_FAIL：InferShape失败。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 单输入场景：IsFinite算子进行inferShape推导，其推导过程与Broadcast算子推导过程一致，可直接复用
ge::graphStatus InferShape4IsFinite(gert::InferShapeContext* context)
{
    return Ops::Base::InferShape4Broadcast(context);
}
IMPL_OP_INFERSHAPE(IsFinite).InferShape(InferShape4IsFinite);

// 多输入场景：多输入broadcast类算子
ge::graphStatus InferShape4MyOp(gert::InferShapeContext* context)
{
    return Ops::Base::InferShape4Broadcast(context, 3); // 3个输入
}
IMPL_OP_INFERSHAPE(MyOp).InferShape(InferShape4MyOp);
```
