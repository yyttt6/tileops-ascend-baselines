# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API| Description|
| --- | --- |
| InitTensor(const int64_t \*viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t \*stride, int64_t offset, aclFormat format, const int64_t \*storageDims, uint64_t storageDimsNum, void \*tensorDataAddr) | Resets the attributes of an `aclTensor` to their initial states.|
| ToString(op::DataType dataType) | Returns the string representation of the specified data type.|
| TypeSize(DataType dataType) | Returns the bit width of the specified data type.|
| ToContiguousStrides(const op::Shape &shape, op::Strides &strides) | Derives the stride information of a contiguous tensor from the given shape.|
| aclStorage() | Constructs an instance of `aclStorage`. `aclStorage` is used to describe a block of on-chip memory.|
| aclStorage(void *addr) | Constructs an instance of `aclStorage`. `aclStorage` is used to describe a block of on-chip memory.|
| aclStorage(bool fromWorkspace) | Constructs an instance of `aclStorage`. `aclStorage` is used to describe a block of on-chip memory.|
| aclStorage(void *addr, bool fromWorkspace) | Constructs an instance of `aclStorage`. `aclStorage` is used to describe a block of on-chip memory.|
| GetAddr() | Obtains the pointer to the on-chip memory address represented by `Storage`.|
| SetAddr(void *addr) | Sets the pointer to the on-chip memory address represented by `Storage`.|
| SetWorkspaceOffset(uint64_t offset) | Sets the offset of the valid data within the device memory when `Storage` represents a workspace block.|
| GetWorkspaceOffset() | Obtains the offset of the valid data within the device memory when `Storage` represents a workspace block.|
| SetFromWorkspace(bool fromWorkspace) | Sets whether `Storage` represents a workspace block.|
| SetFromWorkspace(bool from_workspace) | Sets `aclTensor` to represent a workspace block.|
| IsFromWorkspace() | Checks if `Storage` represents a workspace block.|
| SetExtend(void *extend) | Sets the content of the `Extend` member variable.|
| GetExtend() | Obtains the content of the `Extend` member variable.|
| SetStorageOffset(int64_t offset) | Sets the offset of the valid data within the device memory represented by `Storage`.|
| GetStorageOffset() | Obtains the offset of the valid data within the device memory represented by `Storage`.|
| aclArray() | Constructs an instance of `aclArray`. `aclArray` is used to describe a host-side array.|
| Size() | Obtains the size of an `aclArray`.|
| GetData() | Obtains the pointer to the data area of an `aclArray`.|
| ToString() | Returns the string form of `aclArray` data.|
| aclArray(const T *value, uint64_t size) | Constructs an instance of `aclArray`. `aclArray` is used to describe a host-side array.|
| aclTensor(aclTensor &&other) | This `aclTensor` constructor is forbidden.|
| GetTensor() | Obtains `gert::Tensor` stored in an `aclTensor`.<br> Note: For details, see the description about `gert::Tensor`.|
| GetStorageAddr() | Obtains the pointer to the on-chip memory addressed by `aclStorage` inside an `aclTensor`.|
| GetStorage() | Obtains `aclStorage` inside an `aclTensor`.|
| GetViewOffset() | Obtains `ViewOffset` of an `aclTensor`.|
| SetStorageAddr(void *addr) | Sets the pointer to the on-chip memory addressed by `aclStorage` inside an `aclTensor`.|
| SetViewOffset(int64_t offset) | Sets `ViewOffset` of an `aclTensor`.|
| GetPlacement() | Obtains the type of an `aclTensor`, for example, a device-side tensor or a host-side tensor.|
| aclTensor(const int64_t \*viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t \*stride, int64_t offset, const aclFormat format, const int64_t \*storageDims, uint64_t storageDimsNum, void \*tensorDataAddr) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(op::DataType dataType, op::Format storageFormat, op::Format originFormat) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const op::Shape &shape, op::DataType dataType, op::Format format, void *tensorDataAddr) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat, void *tensorDataAddr) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const op::Shape &shape, op::DataType dataType, op::Format format) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclTensor &other, const op::Shape &shape, int64_t offset) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclIntArray *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclBoolArray *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclFloatArray *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclFp16Array *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclBf16Array *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor() | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const T *value, uint64_t size, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| aclTensor(const aclScalar *value, op::DataType dataType) | Describes a tensor (device-side or host-side tensor). It can describe attributes such as the shape, data type, and format and record data location.|
| IsView() | Checks if an `aclTensor` is a view tensor.|
| SetView(bool is_view) | Sets whether an `aclTensor` is a view tensor.|
| aclTensorList(const aclTensor \*const \*tensors, uint64_t size) | Constructs an instance of `aclTensorList`. `aclTensorList` is an array of `aclTensor` instances.|
| aclTensorList() | Constructs an instance of `aclTensorList`. `aclTensorList` is an array of `aclTensor` instances.|
| aclScalar(const void *data, op::DataType dataType) | Constructs an instance of `aclScalar`. `aclScalar` is a scalar value.|
| aclScalar() | Constructs an instance of `aclScalar`. `aclScalar` is a scalar value.|
| ToFloat() | Converts the scalar stored in an `aclScalar` to a `float` value.|
| ToDouble() | Converts the scalar stored in an `aclScalar` to a `double` value.|
| ToBool() | Converts the scalar stored in an `aclScalar` to a `bool` value.|
| ToInt8() | Converts the scalar stored in an `aclScalar` to an `int8` value.|
| ToInt16() | Converts the scalar stored in an `aclScalar` to an `int16` value.|
| ToInt32() | Converts the scalar stored in an `aclScalar` to an `int32` value.|
| ToInt64() | Converts the scalar stored in an `aclScalar` to an `int64` value.|
| ToUint8() | Converts the scalar stored in an `aclScalar` to a `uint8` value.|
| ToUint16() | Converts the scalar stored in an `aclScalar` to a `uint16` value.|
| ToUint32() | Converts the scalar stored in an `aclScalar` to a `uint32` value.|
| ToUint64() | Converts the scalar stored in an `aclScalar` to a `uint64` value.|
| ToFp16() | Converts the scalar stored in an `aclScalar` to a `float16` value.|
| ToBf16() | Converts the scalar stored in an `aclScalar` to a `bfloat16` value.|
| ToComplex64() | Converts the scalar stored in an `aclScalar` to a `complex64` value.|
| ToComplex128() | Converts the scalar stored in an `aclScalar` to a `complex128` value.|
| aclScalar(int32_t value) | Initializes an `aclScalar` with the specified `int32` type.|
| aclScalar(int64_t value) | Initializes an `aclScalar` with the specified `int64` type.|
| aclScalar(int16_t value) | Initializes an `aclScalar` with the specified `int16` type.|
| aclScalar(int8_t value) | Initializes an `aclScalar` with the specified `int8` type.|
| aclScalar(uint32_t value) | Initializes an `aclScalar` with the specified `uint32` type.|
| aclScalar(uint64_t value) | Initializes an `aclScalar` with the specified `uint64` type.|
| aclScalar(uint16_t value) | Initializes an `aclScalar` with the specified `uint16` type.|
| aclScalar(uint8_t value) | Initializes an `aclScalar` with the specified `uint8` type.|
| aclScalar(float value) | Initializes an `aclScalar` with the specified `float` type.|
| aclScalar(double value) | Initializes an `aclScalar` with the specified `double` type.|
| aclScalar(op::fp16_t value) | Initializes an `aclScalar` with the specified `float16` type.|
| aclScalar(op::bfloat16 value) | Initializes an `aclScalar` with the specified `bfloat16` type.|
| aclScalar(bool value) | Initializes an `aclScalar` with the specified `bool` type.|
| To() | Converts the scalar stored in an `aclScalar` to the data type `T`.|
| ToStr() | Generates a string that describes an `aclScalar`.|
| BFloat16() | Converts the scalar stored in an `aclScalar` to a `bfloat16` value.|
| aclScalarList(const aclScalar \*const \*scalars, uint64_t size) | Constructs an instance of `aclScalarList`. `aclScalarList` is an array of `aclScalar` instances.|
| aclScalarList() | Constructs an instance of `aclScalarList`. `aclScalarList` is an array of `aclScalar` instances.|
| ToString(const aclTensor* t) | Generates a string describing an `aclTensor`. It is a printing function.|
| ToString(const aclTensorList* t) | Generates a string describing an `aclTensorList`. It is a printing function.|
| ToString(const aclIntArray* t) | Generates a string describing an `aclIntArray`. It is a printing function.|
| ToString(const aclBoolArray* t) | Generates a string describing an `aclBoolArray`. It is a printing function.|
| ToString(const aclFloatArray* t) | Generates a string describing an `aclFloatArray`. It is a printing function.|
| ToString(const aclFp16Array* t) | Generates a string describing an `aclFp16Array`. It is a printing function.|
| ToString(const aclBf16Array* t) | Generates a string describing an `aclBf16Array`. It is a printing function.|
| ToString(const aclScalar* t) | Generates a string describing an `aclScalar`. It is a printing function.|
| ToString(const aclScalarList* t) | Generates a string describing an `aclScalarList`. It is a printing function.|
| ToString(const aclTensor& t) | Generates a string describing an `aclTensor&`. It is a printing function.|
| ToString(const aclTensorList& t) | Generates a string describing an `aclTensorList&`. It is a printing function.|
| ToString(const aclIntArray& t) | Generates a string describing an `aclIntArray&`. It is a printing function.|
| ToString(const aclBoolArray& t) | Generates a string describing an `aclBoolArray&`. It is a printing function.|
| ToString(const aclFloatArray& t) | Generates a string describing an `aclFloatArray&`. It is a printing function.|
| ToString(const aclFp16Array& t) | Generates a string describing an `aclFp16Array&`. It is a printing function.|
| ToString(const aclBf16Array& t) | Generates a string describing an `aclBf16Array&`. It is a printing function.|
| ToString(const aclScalar& t) | Generates a string describing an `aclScalar&`. It is a printing function.|
| ToString(const aclScalarList& t) | Generates a string describing an `aclScalarList&`. It is a printing function.|
| ToString(aclDataType dataType) | Generates a string describing an `aclDataType`. It is a printing function.|
