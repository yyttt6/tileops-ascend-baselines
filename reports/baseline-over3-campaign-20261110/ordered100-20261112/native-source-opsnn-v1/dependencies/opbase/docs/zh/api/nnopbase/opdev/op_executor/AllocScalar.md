# AllocScalar

## 功能说明

申请一个aclScalar对象，并对其赋值。通过多个重载函数，用于支持多种数据类型。

## 函数原型

```cpp
aclScalar *AllocScalar(const void *data, op::DataType dataType)
```

```cpp
aclScalar *AllocScalar(float value)
```

```cpp
aclScalar *AllocScalar(double value)
```

```cpp
aclScalar *AllocScalar(op::fp16_t value)
```

```cpp
aclScalar *AllocScalar(op::bfloat16 value)
```

```cpp
aclScalar *AllocScalar(int32_t value)
```

```cpp
aclScalar *AllocScalar(int64_t value)
```

```cpp
aclScalar *AllocScalar(int16_t value)
```

```cpp
aclScalar *AllocScalar(int8_t value)
```

```cpp
aclScalar *AllocScalar(uint32_t value)
```

```cpp
aclScalar *AllocScalar(uint64_t value)
```

```cpp
aclScalar *AllocScalar(uint16_t value)
```

```cpp
aclScalar *AllocScalar(uint8_t value)
```

```cpp
aclScalar *AllocScalar(bool value)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data | 输入 | 源数据指针。 |
| dataType | 输入 | 源数据的数据类型。 |
| value | 输入 | 将aclScalar的内容指定为value。 |

## 返回值说明

申请到的aclScalar对象，申请失败返回nullptr。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 初始化一个值为5，数据类型为int64的aclScalar对象
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    aclScalar *scalar = executor->AllocScalar(val);
    scalar = executor->AllocScalar(&val, DT_INT64);
}
```
