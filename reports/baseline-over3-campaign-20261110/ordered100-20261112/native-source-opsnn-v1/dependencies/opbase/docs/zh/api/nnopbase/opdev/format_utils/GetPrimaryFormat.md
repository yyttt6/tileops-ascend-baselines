# GetPrimaryFormat

## 功能说明

获取根format，去除关于C0的信息，只保留单纯的format信息。例如，将FORMAT\_FRACTAL\_Z\_C04转为FORMAT\_FRACTAL\_Z。

## 函数原型

```cpp
int32_t GetPrimaryFormat(int32_t format)
```

```cpp
Format GetPrimaryFormat(Format format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 原始目标数据格式。 |

## 返回值说明

去除C0信息后的格式。

## 约束说明

无

## 调用示例

```cpp
// 判断当input的storage format是fractal z时，返回
void Func(const aclTensor *input) {
    if (GetPrimaryFormat(input->GetStorageFormat()) == FORMAT_FRACTAL_Z){
        return;
    }
}
```
