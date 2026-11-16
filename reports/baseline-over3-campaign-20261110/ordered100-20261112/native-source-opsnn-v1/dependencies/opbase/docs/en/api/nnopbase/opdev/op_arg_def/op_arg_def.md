# op\_arg\_def

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API Definition| Description|
| --- | --- |
| VisitTupleElem(const F &func, const std::tuple<Ts...> &tp) | Traverses all elements of `OpArgBase` and calls `func`.|
| VisitTupleElemNoReturn(const F &func, const std::tuple<Ts...> &tp) | Traverses all elements of `OpArgBase` and calls `func`.|
| VisitTupleElemAt(size_t idx, const F &func, const std::tuple<Ts...> &tp) | Calls `func` for the element with the specified `idx` in `OpArgBase`.|
| OpArgBase(const std::tuple<T...> &arg) | Indicates the `OpArgBase` constructor.|
| OpArgBase(std::tuple<T...> &&arg) | Indicates the `OpArgBase` constructor.|
| OpArgBase() | Indicates the `OpArgBase` constructor.|
| Size() | Obtains the number of elements in `OpArgBase`.|
| VisitBy(const F &func) | Traverses all elements of `OpArgBase` and calls `func`.|
| VisitTupleElem(func, arg_) | Traverses all elements of `OpArgBase` and calls `func`.|
| VisitByNoReturn(const F &func) | Calls `func` for the element with the specified `idx` in `OpArgBase`.|
| VisitAt(size_t idx, const F &func) | Calls `func` for the element with the specified `idx` in `OpArgBase`.|
| OpArgTypeStr(int argType) | Converts `argType` to a string.|
| ExtractOpArgType(const T &t, const Ts &...ts) | Obtains the elements of the V type from `ts`.|
| ExtractOpArgTypeTuple(const Tuple &t) | Obtains the elements of the V type from `t`.|
| OpArgValue() | Indicates the `OpArgValue` constructor.|
| OpArgValue(const aclTensor *value) | Constructs `OpArgValue` using a value of the `const aclTensor` type and assigns a value to it.|
| OpArgValue(aclTensor *value) | Constructs `OpArgValue` using a value of the `aclTensor` type and assigns a value to it.|
| OpArgValue(const aclTensorList *value) | Constructs `OpArgValue` using a value of the `const aclTensorList` type and assigns a value to it.|
| OpArgValue(aclTensorList *value) | Constructs `OpArgValue` using a value of the `aclTensorList` type and assigns a value to it.|
| OpArgValue(std::string *value) | Constructs `OpArgValue` using a value of the `string` type and assigns a value to it.|
| OpArgValue(const std::string *value) | Constructs `OpArgValue` using a value of the `const string` type and assigns a value to it.|
| OpArgValue(std::string &value) | Constructs `OpArgValue` using a value of the `string&` type and assigns a value to it.|
| OpArgValue(const std::string &value) | Constructs `OpArgValue` using a value of the `const string&` type and assigns a value to it.|
| OpArgValue(const char *value) | Constructs `OpArgValue` using a value of the `const aclTensor` type and assigns a value to it.|
| OpArgValue(char *value) | Constructs `OpArgValue` using a value of the `const char` type and assigns a value to it.|
| OpArgValue(std::vector<std::tuple<void *, const aclTensor*>> *value) | Constructs `OpArgValue` using a value of the `vector` type and assigns a value to it.|
| OpArgValue(const std::vector<std::tuple<void*, const aclTensor*>> *value) | Constructs `OpArgValue` using a value of the `const vector` type and assigns a value to it.|
| OpArgValue(double value) | Constructs `OpArgValue` using a value of the `double` type and assigns a value to it.|
| OpArgValue(uint32_t value) | Constructs `OpArgValue` using a value of the `uint32` type and assigns a value to it.|
| OpArgValue(int32_t value) | Constructs `OpArgValue` using a value of the `int32` type and assigns a value to it.|
| OpArgValue(float value) | Constructs `OpArgValue` using a value of the `float` type and assigns a value to it.|
| OpArgValue(const bool value) | Constructs `OpArgValue` using a value of the `const bool` type and assigns a value to it.|
| OpArgValue(const DataType value) | Constructs `OpArgValue` using a value of the `const DataType` type and assigns a value to it.|
| OpArgValue(aclScalar *value) | Constructs `OpArgValue` using a value of the `aclScalar` type and assigns a value to it.|
| OpArgValue(const aclScalar *value) | Constructs `OpArgValue` using a value of the `const aclScalar` type and assigns a value to it.|
| OpArgValue(aclIntArray *value) | Constructs `OpArgValue` using a value of the `aclIntArray` type and assigns a value to it.|
| OpArgValue(const aclIntArray *value) | Constructs `OpArgValue` using a value of the `const aclIntArray` type and assigns a value to it.|
| OpArgValue(aclFloatArray *value) | Constructs `OpArgValue` using a value of the `aclFloatArray` type and assigns a value to it.|
| OpArgValue(const aclFloatArray *value) | Constructs `OpArgValue` using a value of the `const aclFloatArray` type and assigns a value to it.|
| OpArgValue(aclBoolArray *value) | Constructs `OpArgValue` using a value of the `aclBoolArray` type and assigns a value to it.|
| OpArgValue(const aclBoolArray *value) | Constructs `OpArgValue` using a value of the `const aclBoolArray` type and assigns a value to it.|
| OpArgValue(op::OpImplMode value) | Constructs `OpArgValue` using a value of the `op::OpImplMode` type and assigns a value to it.|
| OpArgValue(const T &value) | Constructs `OpArgValue` using a value of the `const T& type` and assigns a value to it.|
| OpArg() | Indicates the `OpArg` constructor.|
| OpArgList() | Indicates the `OpArgList` constructor.|
| OpArgList(OpArg *args\_, size_t count_) | Creates an `OpArgList` with the given `args`.|
| OpArgContext() | Indicates the `OpArgContext` constructor.|
| GetOpArg(OpArgDef type) | Obtains an `OpArgList` with the given type.|
| ContainsOpArgType(OpArgDef type) | Checks whether `OpArgContext` contains an `OpArgList` of the execution type.|
| AppendOpWorkspaceArg(aclTensorList *tensorList) | Adds the given `TensorList` to `OpArgContext` as a workspace parameter.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclTensor *tensor, OpArg*&currArg) | Adds the given `const aclTensor` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclTensor *tensor, OpArg*&currArg) | Adds the given `aclTensor` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclTensorList *tensorList, OpArg*&currArg) | Adds the given `const aclTensorList` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclTensorList *tensorList, OpArg*&currArg) | Adds the given `aclTensorList` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, [[maybe_unused]] const std::nullptr_t tensor, OpArg *&currArg) | Adds the given `const std::nullptr_t` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const bool value, OpArg *&currArg) | Adds the given `const bool` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const DataType value, OpArg *&currArg) | Adds the given `const DataType` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclScalar *value, OpArg*&currArg) | Adds the given `aclScalar` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclScalar *value, OpArg*&currArg) | Adds the given `const aclScalar` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclIntArray *value, OpArg*&currArg) | Adds the given `aclIntArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclIntArray *value, OpArg*&currArg) | Adds the given `const aclIntArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclFloatArray *value, OpArg*&currArg) | Adds the given `aclFloatArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclFloatArray *value, OpArg*&currArg) | Adds the given `const aclFloatArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, aclBoolArray *value, OpArg*&currArg) | Adds the given `aclBoolArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const aclBoolArray *value, OpArg*&currArg) | Adds the given `const aclBoolArray` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, std::string *value, OpArg*&currArg) | Adds the given `string` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const std::string *value, OpArg*&currArg) | Adds the given `const string` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, std::string &value, OpArg *&currArg) | Adds the given `string &` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const std::string &value, OpArg *&currArg) | Adds the given `const string &` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const char *value, OpArg*&currArg) | Adds the given `const char` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, char *value, OpArg*&currArg) | Adds the given `char` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, double value, OpArg *&currArg) | Adds the given `double` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, float value, OpArg *&currArg) | Adds the given `float` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, int32_t value, OpArg *&currArg) | Adds the given `int32` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, uint32_t value, OpArg *&currArg) | Adds the given `uint32` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, op::OpImplMode value, OpArg *&currArg) | Adds the given `op::OpImplMode` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, std::vector<std::tuple<void*, const aclTensor\*>> &value, OpArg*&currArg) | Adds the given `vector` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, const std::vector<std::tuple<void*, const aclTensor\*>> &value, OpArg*&currArg) | Adds the given `const vector` parameter to `OpArgList`.|
| AppendOpArg([[maybe_unused]] size_t idx, T value, OpArg *&currArg) | Adds the given `T` parameter to `OpArgList`.|
| OpArgContextSize(const T &t, const Ts &...ts) | Calculates the size of `OpArgContext` based on the given parameters.|
| OpArgContextInit(OpArgContext &ctx, OpArg *&currArg, const T &t, const Ts &...ts) | Initializes `OpArgContext` based on the given parameters.|
| MakeOpArgContext(const Ts &...ts) | Creates an `OpArgContext` based on the given parameters.|
| GetOpArgContext(const Ts &...ts) | Obtains `OpArgContext` based on the given parameters.|
| DestroyOpArgContext(OpArgContext *ctx) | Destroys the specified `OpArgContext`.|
| Allocated(size_t size) | Allocates memory of the specified size.|
| DeAllocated(void *addr) | Frees the memory at the address specified by `addr`.|
