# SetViewShape

## 功能说明

设置aclTensor的ViewShape。ViewShape表示aclTensor的逻辑shape，可由[GetViewShape](GetViewShape.md)获取。

## 函数原型

```cpp
void SetViewShape(const op::Shape &shape)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 数据类型为op::Shape（即gert::Shape），记录了一组shape信息，例如一个三维shape：[10, 20, 30]。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 将input的ViewShape设置为[1, 2, 3, 4, 5]
void Func(aclTensor *input) {
    gert::Shape newShape;
    for (int64_t i = 1; i <= 5; i++) {
        newShape.AppendDim(i);
    }
    input->SetViewShape(newShape);
}
```
