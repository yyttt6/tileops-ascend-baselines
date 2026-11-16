# ConvertToTensor

## 功能说明

将不同数据类型的host侧数据，转为一个host侧aclTensor对象。

## 函数原型

```cpp
const aclTensor *ConvertToTensor(const aclIntArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclBoolArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclFloatArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclFp16Array *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclBf16Array *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclScalar *value, DataType dataType)
```

```cpp
template<typename T>
const aclTensor *ConvertToTensor(const T *value, uint64_t size, DataType dataType)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | host侧源数据。 |
| size | 输入 | 源数据的元素个数。 |
| dataType | 输入 | 将源数据转换为指定数据类型后，写入aclTensor。 |

## 返回值说明

返回转换后的host侧tensor。

## 约束说明

入参指针不能为空。

### 各接口参数数据类型约束

#### 接口1: `ConvertToTensor(const aclIntArray *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclIntArray类型，内部存储int64_t数据 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

#### 接口2: `ConvertToTensor(const aclBoolArray *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclBoolArray类型，内部存储bool数据 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

#### 接口3: `ConvertToTensor(const aclFloatArray *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclFloatArray类型，内部存储float数据 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

#### 接口4: `ConvertToTensor(const aclFp16Array *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclFp16Array类型，内部存储fp16_t数据 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

#### 接口5: `ConvertToTensor(const aclBf16Array *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclBf16Array类型，内部存储bfloat16数据 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

#### 接口6: `ConvertToTensor(const aclScalar *value, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | aclScalar类型，支持的数据类型包括：DT_BOOL、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FLOAT、DT_DOUBLE、DT_FLOAT16、DT_BF16、DT_COMPLEX64、DT_COMPLEX128、DT_FLOAT8_E5M2、DT_FLOAT8_E4M3FN、DT_FLOAT8_E8M0、DT_FLOAT6_E3M2、DT_FLOAT6_E2M3、DT_FLOAT4_E2M1、DT_FLOAT4_E1M2、DT_HIFLOAT8 |
| dataType | 支持的数据类型：DT_BOOL、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FLOAT、DT_DOUBLE、DT_FLOAT16、DT_BF16、DT_COMPLEX64、DT_COMPLEX128、DT_FLOAT8_E5M2、DT_FLOAT8_E4M3FN、DT_FLOAT8_E8M0、DT_FLOAT6_E3M2、DT_FLOAT6_E2M3、DT_FLOAT4_E2M1、DT_FLOAT4_E1M2、DT_HIFLOAT8 |

#### 接口7: `ConvertToTensor(const T *value, uint64_t size, DataType dataType)`

| 参数 | 数据类型约束 |
| --- | --- |
| value | 模板类型T支持：int64_t、uint64_t、int32_t、uint32_t、int8_t、uint8_t、int16_t、uint16_t、float、double、bool、char、op::bfloat16、op::fp16_t |
| size | uint64_t类型，表示元素个数 |
| dataType | 支持的数据类型：DT_FLOAT、DT_FLOAT16、DT_BF16、DT_INT8、DT_INT16、DT_UINT16、DT_UINT8、DT_INT32、DT_INT64、DT_UINT32、DT_UINT64、DT_BOOL、DT_DOUBLE |

## 调用示例

```cpp
// 分别将一个aclScalar和int64_t转为host侧tensor
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    aclScalar *scalar = executor->AllocScalar(val);
    const aclTensor *tensor = executor->ConvertToTensor(scalar, DT_INT64);
    tensor = executor->ConvertToTensor(&val, 1, DT_INT64);
}
```
