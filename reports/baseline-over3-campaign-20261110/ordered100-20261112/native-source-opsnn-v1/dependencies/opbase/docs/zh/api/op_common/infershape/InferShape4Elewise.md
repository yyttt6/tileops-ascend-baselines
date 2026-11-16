# InferShape4Elewise

## 功能说明

图模式场景下，elewise类算子的InferShape方法（用于推导输出张量的形状shape）。

## 函数原型

```cpp
ge::graphStatus InferShape4Elewise(gert::InferShapeContext *context)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输出 | GE传入的InferShape上下文。 |

## 返回值说明

返回类型为ge::graphStatus：

- ge::GRAPH\_SUCCESS：InferShape成功。
- ge::GRAPH\_FAIL：InferShape失败。

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// IsFinite算子进行inferShape推导，其推导过程与Elewise算子推导过程一致，可直接复用
ge::graphStatus InferShape4IsFinite(gert::InferShapeContext* context)
{
    return Ops::Base::InferShape4Elewise(context);
}
// IsFinite算子及其推导函数注册到GE
IMPL_OP_INFERSHAPE(IsFinite).InferShape(InferShape4IsFinite);
}
```
