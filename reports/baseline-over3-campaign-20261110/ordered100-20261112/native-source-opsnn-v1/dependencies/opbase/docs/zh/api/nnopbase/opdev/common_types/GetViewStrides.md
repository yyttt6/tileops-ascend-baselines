# GetViewStrides

## 功能说明

获取aclTensor的ViewStrides。

## 函数原型

```cpp
FVector<int64_t> GetViewStrides()
```

## 参数说明

无

## 返回值说明

返回一个FVector对象，里面存放aclTensor各个维度的stride大小。

## 约束说明

无

## 调用示例

```cpp
// 获取input的View Stride并依次打印出每一维的stride大小
void Func(const aclTensor *input) {
    auto strides = input->GetViewStrides();
    for (int64_t stride : strides) {
        std::cout << stride << std::endl;
    }
}
```
