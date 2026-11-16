# CheckOverflows

## 功能说明

校验aclScalar中保存的标量，数值是否有溢出。

## 函数原型

```cpp
template<typename to> bool CheckOverflows()
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| to | 输入 | 判断当前是aclScalar转为to表示的数据类型时，是否会溢出。 |

## 返回值说明

如果存在溢出，则返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 判断input转为fp16和int16是否会发生溢出
void Func(const aclScalar *input) {
    if (input->CheckOverflows<fp16_t>()) {
        return;
    }
    if (input->CheckOverflows<int16_t>()) {
        return;
    }
}
```
