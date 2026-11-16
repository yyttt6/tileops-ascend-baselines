# Numel

## 功能说明

获取aclTensor的总元素个数。

## 函数原型

```cpp
int64_t Numel()
```

## 参数说明

无

## 返回值说明

返回aclTensor的总元素个数。

## 约束说明

无

## 调用示例

```cpp
// 获取input的总元素个数
void Func(const aclTensor *input) {
    int64_t num = input->Numel();
}
```
