# 预留接口

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口名称 | 功能说明 |
| --- | --- |
| InitTensor(const int64_t \*viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t \*stride, int64_t offset, aclFormat format, const int64_t \*storageDims, uint64_t storageDimsNum, void \*tensorDataAddr) | 重置一个aclTensor的各项属性。 |
| ToString(op::DataType dataType) | 获取数据类型对应字符串表示。 |
| TypeSize(DataType dataType) | 获取数据类型位宽。 |
| ToContiguousStrides(const op::Shape &shape, op::Strides &strides) | 根据shape推导连续tensor的stride。 |
| aclStorage() | aclStorage的构造函数，aclStorage用于描述一块片上内存空间。 |
| aclStorage(void *addr) | aclStorage的构造函数，aclStorage用于描述一块片上内存空间。 |
| aclStorage(bool fromWorkspace) | aclStorage的构造函数，aclStorage用于描述一块片上内存空间。 |
| aclStorage(void *addr, bool fromWorkspace) | aclStorage的构造函数，aclStorage用于描述一块片上内存空间。 |
| GetAddr() | 获取Storage表示的片上内存地址指针。 |
| SetAddr(void *addr) | 设置Storage表示的片上内存地址指针。 |
| SetWorkspaceOffset(uint64_t offset) | 当Storage表示一块workspace时，设置其有效数据在device存储内存的偏移。 |
| GetWorkspaceOffset() | 当Storage表示一块workspace时，获取其有效数据在device存储内存的偏移。 |
| SetFromWorkspace(bool fromWorkspace) | aclStorage::SetFromWorkspace，设置Storage是否表示一块workspace。 |
| SetFromWorkspace(bool from_workspace) | aclTensor::SetFromWorkspace，设置aclTensor表示一块workspace。 |
| IsFromWorkspace() | 获取Storage是否表示一块workspace。 |
| SetExtend(void *extend) | 设置extend成员变量内容。 |
| GetExtend() | 获取extend成员变量内容。 |
| SetStorageOffset(int64_t offset) | 设置Storage有效数据在指向device存储内存上的偏移。 |
| GetStorageOffset() | 获取Storage有效数据在指向device存储内存上的偏移。 |
| aclArray() | aclArray构造函数，aclArray用于描述一个host侧数组。 |
| Size() | 获取aclArray的大小。 |
| GetData() | 获取aclArray的数据区指针。 |
| ToString() | 获取aclArray内容的打印字符串。 |
| aclArray(const T *value, uint64_t size) | aclArray构造函数，aclArray用于描述一个host侧数组。 |
| aclTensor(aclTensor &&other) | 禁止aclTensor的该类构造函数。 |
| GetTensor() | 获取aclTensor中保存的gert::Tensor。 |
| GetStorageAddr() | 获取aclTensor内部aclStorage指向的片上内存指针。 |
| GetStorage() | 获取aclTensor内部的aclStorage。 |
| GetViewOffset() | 获取aclTensor的ViewOffset。 |
| SetStorageAddr(void *addr) | 设置aclTensor内部aclStorage指向的片上内存指针。 |
| SetViewOffset(int64_t offset) | 设置aclTensor的ViewOffset。 |
| GetPlacement() | 获取aclTensor的类型，例如device侧tensor和host侧tensor。 |
| aclTensor(const int64_t \*viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t \*stride, int64_t offset, const aclFormat format, const int64_t \*storageDims, uint64_t storageDimsNum, void \*tensorDataAddr) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(op::DataType dataType, op::Format storageFormat, op::Format originFormat) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const op::Shape &shape, op::DataType dataType, op::Format format, void *tensorDataAddr) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat, void *tensorDataAddr) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const op::Shape &shape, op::DataType dataType, op::Format format) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclTensor &other, const op::Shape &shape, int64_t offset) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclIntArray *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclBoolArray *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclFloatArray *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclFp16Array *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclBf16Array *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor() | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const T *value, uint64_t size, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| aclTensor(const aclScalar *value, op::DataType dataType) | aclTensor用于描述一块张量，包括device侧tensor与host侧tensor两种。可以描述tensor的shape、dtype、format等属性，并记录数据所处位置。 |
| SetFloat8E5M2Data(const op::Float8E5M2 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float8_e5m2类型的数据。 |
| SetFloat8E4M3FNData(const op::Float8E4M3FN \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float8_e4m3fn类型的数据。 |
| SetFloat8E8M0Data(const op::Float8E8M0 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float8_e8m0类型的数据。 |
| SetFloat6E3M2Data(const op::Float6E3M2 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float6_e3m2类型的数据。 |
| SetFloat6E2M3Data(const op::Float6E2M3 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float6_e2m3类型的数据。 |
| SetFloat4E2M1Data(const op::Float4E2M1 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float4_e2m1类型的数据。 |
| SetFloat4E1M2Data(const op::Float4E1M2 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中float4_e1m2类型的数据。 |
| SetHiFloat4Data(const op::HiFloat4 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中hifloat4类型的数据。 |
| SetHiFloat8Data(const op::HiFloat8 \*value, uint64_t size, op::DataType dataType) | 设置aclTensor中hifloat8类型的数据。 |
| IsView() | 获取aclTensor是否为view类tensor。 |
| SetView(bool is_view) | 设置aclTensor是否为view类tensor。 |
| aclTensorList(const aclTensor \*const \*tensors, uint64_t size) | aclTensorList表示一个aclTensor数组，该函数为其构造函数。 |
| aclTensorList() | aclTensorList表示一个aclTensor数组，该函数为其构造函数。 |
| aclScalar(const void *data, op::DataType dataType) | aclScalar表示一个标量数字，该函数为其构造函数。 |
| aclScalar() | aclScalar表示一个标量数字，该函数为其构造函数。 |
| ToFloat() | 将aclScalar中保存的标量，转为指定的数据类型float。 |
| ToDouble() | 将aclScalar中保存的标量，转为指定的数据类型double。 |
| ToBool() | 将aclScalar中保存的标量，转为指定的数据类型bool。 |
| ToInt8() | 将aclScalar中保存的标量，转为指定的数据类型int8_t。 |
| ToInt16() | 将aclScalar中保存的标量，转为指定的数据类型int16_t。 |
| ToInt32() | 将aclScalar中保存的标量，转为指定的数据类型int32_t。 |
| ToInt64() | 将aclScalar中保存的标量，转为指定的数据类型int64_t。 |
| ToUint8() | 将aclScalar中保存的标量，转为指定的数据类型uint8_t。 |
| ToUint16() | 将aclScalar中保存的标量，转为指定的数据类型uint16_t。 |
| ToUint32() | 将aclScalar中保存的标量，转为指定的数据类型uint32_t。 |
| ToUint64() | 将aclScalar中保存的标量，转为指定的数据类型uint64_t。 |
| ToFp16() | 将aclScalar中保存的标量，转为指定的数据类型float16。 |
| ToBf16() | 将aclScalar中保存的标量，转为指定的数据类型bfloat16。 |
| ToFloat8E5M2() | 将aclScalar中保存的标量，转为指定的数据类型float8_e5m2。 |
| ToFloat8E4M3FN() | 将aclScalar中保存的标量，转为指定的数据类型float8_e4m3fn。 |
| ToFloat8E8M0() | 将aclScalar中保存的标量，转为指定的数据类型float8_e8m0。 |
| ToFloat6E3M2() | 将aclScalar中保存的标量，转为指定的数据类型float6_e3m2。 |
| ToFloat6E2M3() | 将aclScalar中保存的标量，转为指定的数据类型float6_e2m3。 |
| ToFloat4E2M1() | 将aclScalar中保存的标量，转为指定的数据类型float4_e2m1。 |
| ToFloat4E1M2() | 将aclScalar中保存的标量，转为指定的数据类型float4_e1m2。 |
| ToHiFloat4() | 将aclScalar中保存的标量，转为指定的数据类型hifloat4。 |
| ToHiFloat8() | 将aclScalar中保存的标量，转为指定的数据类型hifloat8。 |
| ToComplex64() | 将aclScalar中保存的标量，转为指定的数据类型complex64。 |
| ToComplex128() | 将aclScalar中保存的标量，转为指定的数据类型complex128。 |
| aclScalar(int32_t value) | 初始化aclScalar对象，指定为int32_t类型。 |
| aclScalar(int64_t value) | 初始化aclScalar对象，指定为int64_t类型。 |
| aclScalar(int16_t value) | 初始化aclScalar对象，指定为int16_t类型。 |
| aclScalar(int8_t value) | 初始化aclScalar对象，指定为int8_t类型。 |
| aclScalar(uint32_t value) | 初始化aclScalar对象，指定为uint32_t类型。 |
| aclScalar(uint64_t value) | 初始化aclScalar对象，指定为uint64_t类型。 |
| aclScalar(uint16_t value) | 初始化aclScalar对象，指定为uint16_t类型。 |
| aclScalar(uint8_t value) | 初始化aclScalar对象，指定为uint8_t类型。 |
| aclScalar(float value) | 初始化aclScalar对象，指定为float类型。 |
| aclScalar(double value) | 初始化aclScalar对象，指定为double类型。 |
| aclScalar(op::fp16_t value) | 初始化aclScalar对象，指定为float16类型。 |
| aclScalar(op::bfloat16 value) | 初始化aclScalar对象，指定为bfloat16类型。 |
| aclScalar(bool value) | 初始化aclScalar对象，指定为bool类型。 |
| aclScalar(op::Float8E5M2 value) | 初始化aclScalar对象，指定为float8_e5m2类型。 |
| aclScalar(op::Float8E4M3FN value) | 初始化aclScalar对象，指定为float8_e4m3fn类型。 |
| aclScalar(op::Float8E8M0 value) | 初始化aclScalar对象，指定为float8_e8m0类型。 |
| aclScalar(op::Float6E3M2 value) | 初始化aclScalar对象，指定为float6_e3m2类型。 |
| aclScalar(op::Float6E2M3 value) | 初始化aclScalar对象，指定为float6_e2m3类型。 |
| aclScalar(op::Float4E2M1 value) | 初始化aclScalar对象，指定为float4_e2m1类型。 |
| aclScalar(op::Float4E1M2 value) | 初始化aclScalar对象，指定为float4_e1m2类型。 |
| aclScalar(op::HiFloat4 value) | 初始化aclScalar对象，指定为hifloat4类型。 |
| aclScalar(op::HiFloat8 value) | 初始化aclScalar对象，指定为hifloat8类型。 |
| To() | 将aclScalar中保存的标量，转为T数据类型。 |
| ToStr() | 生成描述aclScalar的字符串。 |
| BFloat16() | 将aclScalar中保存的标量，转为bfloat16。 |
| v_t() | aclScalar内部联合体v_t的默认构造函数。 |
| aclScalarList(const aclScalar \*const \*scalars, uint64_t size) | aclScalarList表示一个aclScalar数组，该函数为其构造函数。 |
| aclScalarList() | aclScalarList表示一个aclScalar数组，该函数为其构造函数。 |
| ToString(const aclTensor* t) | 打印函数，生成可以描述aclTensor对象的字符串。 |
| ToString(const aclTensorList* t) | 打印函数，生成可以描述aclTensorList对象的字符串。 |
| ToString(const aclIntArray* t) | 打印函数，生成可以描述aclIntArray对象的字符串。 |
| ToString(const aclBoolArray* t) | 打印函数，生成可以描述aclBoolArray对象的字符串。 |
| ToString(const aclFloatArray* t) | 打印函数，生成可以描述aclFloatArray对象的字符串。 |
| ToString(const aclFp16Array* t) | 打印函数，生成可以描述aclFp16Array对象的字符串。 |
| ToString(const aclBf16Array* t) | 打印函数，生成可以描述aclBf16Array对象的字符串。 |
| ToString(const aclScalar* t) | 打印函数，生成可以描述aclScalar对象的字符串。 |
| ToString(const aclScalarList* t) | 打印函数，生成可以描述aclScalarList对象的字符串。 |
| ToString(const aclTensor& t) | 打印函数，生成可以描述aclTensor&对象的字符串。 |
| ToString(const aclTensorList& t) | 打印函数，生成可以描述aclTensorList&对象的字符串。 |
| ToString(const aclIntArray& t) | 打印函数，生成可以描述aclIntArray&对象的字符串。 |
| ToString(const aclBoolArray& t) | 打印函数，生成可以描述aclBoolArray&对象的字符串。 |
| ToString(const aclFloatArray& t) | 打印函数，生成可以描述aclFloatArray&对象的字符串。 |
| ToString(const aclFp16Array& t) | 打印函数，生成可以描述aclFp16Array&对象的字符串。 |
| ToString(const aclBf16Array& t) | 打印函数，生成可以描述aclBf16Array&对象的字符串。 |
| ToString(const aclScalar& t) | 打印函数，生成可以描述aclScalar&对象的字符串。 |
| ToString(const aclScalarList& t) | 打印函数，生成可以描述aclScalarList&对象的字符串。 |
| ToString(aclDataType dataType) | 打印函数，生成可以描述aclDataType对象的字符串。 |
