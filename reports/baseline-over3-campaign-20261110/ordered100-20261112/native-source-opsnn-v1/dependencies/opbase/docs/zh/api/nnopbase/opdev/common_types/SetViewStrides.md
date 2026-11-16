# SetViewStrides

## 功能说明

设置aclTensor的ViewStrides。

## 函数原型

```cpp
SetViewStrides(const op::Strides &strides)
```

```cpp
SetViewStrides(op::Strides &&strides)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| strides | 输入 | 类型为FVector<int64_t>，表示aclTensor每一维的stride大小，取值不支持负数。 |

## 返回值说明

void

## 约束说明

无

## 调用示例

```cpp
// 将input最后一维的stride扩大8倍。
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    strides[strides.size() - 1] *= 8;
    input->SetViewStrides(strides);
}
```
