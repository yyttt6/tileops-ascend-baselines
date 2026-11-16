# aicpu\_task

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| PrintAicpuAllTimeStampInfo(const char *opType) | Prints the system timestamp when an AI CPU task is executed.|
| AppendTensor(aclOpExecutor *executor, const aclTensor*arg, V &l) | Inserts an `aclTensor` pointer into a container. This is a template function.|
| AppendTensor(aclOpExecutor *executor, const aclScalar*arg, V &l) | Inserts an `aclScalar` pointer into a container. This is a template function.|
| AppendTensor(aclOpExecutor *executor, const aclIntArray*arg, V &l) | Inserts an `aclIntArray` pointer into a container. This is a template function.|
| AppendTensor(aclOpExecutor *executor, const aclTensorList*arg, V &l) | Inserts an `aclTensorList` pointer into a container. This is a template function.|
| CreateTensorListImpl(aclOpExecutor *executor, OpArg &arg, TensorList &l) | Creates a tensor list from input `args`. This is a template function.|
| CreateTensorList(aclOpExecutor *executor, OpArgList &t, TensorList &l) | Creates a tensor list.|
| Append1Byte(uint8_t *buf, uint8_t src) | Appends one byte to the specified buffer.|
| AppendAttrForKey(const V &value, uint8_t *&key, size_t &keyLen) | Appends attribute information to the task key field. This is a template function.|
| AppendAttrForKey(const std::string &value, uint8_t *&key, size_t &keyLen) | Appends `const string &` attribute information to the task key field.|
| AppendAttrForKey(const std::string *value, uint8_t*&key, size_t &keyLen) | Appends `const string` pointer attribute information to the task key field.|
| AppendAttrForKey(std::string *value, uint8_t*&key, size_t &keyLen) | Appends `string` pointer attribute information to the task key field.|
| AppendAttrForKey(const std::vector\<V\> &value, uint8_t *&key, size_t &keyLen) | Appends `vector` attribute information to the task key field.|
| AppendAttrForKey(const aclIntArray *value, uint8_t*&key, size_t &keyLen) | Appends `const aclIntArray` attribute information to the task key field.|
| AppendAttrForKey(aclIntArray *value, uint8_t*&key, size_t &keyLen) | Appends `aclIntArray` attribute information to the task key field.|
| AppendAttrForKey(const aclFloatArray *value, uint8_t*&key, size_t &keyLen) | Appends `const aclFloatArray` attribute information to the task key field.|
| AppendAttrForKey(aclFloatArray *value, uint8_t*&key, size_t &keyLen) | Appends `aclFloatArray` attribute information to the task key field.|
| AppendAttrForKey(const aclBoolArray *value, uint8_t*&key, size_t &keyLen) | Appends `const aclBoolArray` attribute information to the task key field.|
| AppendAttrForKey(aclBoolArray *value, uint8_t*&key, size_t &keyLen) | Appends `aclBoolArray` attribute information to the task key field.|
| AddAicpuAttr(const aclIntArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(aclIntArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(const aclFloatArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(aclFloatArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(const aclBoolArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(aclBoolArray *value, const std::string &attrName, AicpuAttrs &attrs) | Adds an AI CPU attribute field.|
| AddAicpuAttr(const V &value, const std::string &attrName, AicpuAttrs &attrs) | Adds a simple data type attribute into the AI CPU attribute map.|
| GetTid() | Obtains the ID of the current working thread.|
| aclnnAicpuFinalize() | Deinitializes the AI CPU module.|
| CreatAicpuKernelLauncher(uint32_t opType, op::internal::AicpuTaskSpace &space, aclOpExecutor *executor, const FVector\<std::string\> &attrNames, op::OpArgContext*args) | Creates an AI CPU task delivery object.|
| AicpuTask() | Constructs a default instance of the `AicpuTask` class.|
| AicpuTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | Constructs an instance of the `AicpuTask` class (with input parameters).|
| AicpuTfTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | Constructs a TensorFlow operator task for the AI CPU framework.|
| aclnnStatus Init(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, const AicpuAttrs &attrs) | Initializes an operator framework task.|
| aclnnStatus Run(aclOpExecutor *executor, aclrtStream stream) | Runs an operator framework task.|
| AicpuTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | Constructs an AI CPU task.|
| AicpuCCTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | Constructs a CANN operator task for the AI CPU framework.|
| aclnnStatus SetIoTensors(aclOpExecutor *executor, op::OpArgContext*args) | Refreshes the input and output addresses of a task.|
| void SetSpace(void *space) | Sets the container where a task is located.|
| void SetVisit(bool visit) | Sets whether a task is occupied.|
| AicpuTaskSpace(const std::string &opType,  const ge::UnknowShapeOpType unknownType = ge::DEPEND_IN_SHAPE,  const bool isTf = false) | Constructs an AI CPU task container.|
| AicpuTask *FindTask(aclOpExecutor*executor, op::OpArgContext *args, const FVector<const aclTensor*> &inputs) | Checks if a task can be reused.|
| AicpuTask *GetOrCreateTask(aclOpExecutor*executor, const FVector\<std::string\> &attrNames, op::OpArgContext *args) | Obtains a new or reused task.|
| void SetRef(const size_t index, const bool isInput = true) | Sets the specified index as a reference-type input.|
| bool IsRef(const size_t index, const bool isInput = true) const | Checks if the specified index is a reference-type input.|
| uint64_t CalcHostInputDataSize(const FVector<const aclTensor *> &inputs, size_t alignBytes) const | Calculates the total size of input data stored on the host side.|
| uint64_t CalcDeviceCacheSize(const FVector<const aclTensor *> &inputs, std::unique_ptr\<AicpuTask\> &aicpuTask) const | Calculates the size of the memory reserved on the device side.|
| void Clear() | Clears cached tasks.|
| static size_t GenHashBinary(const uint8_t *addr, uint32_t len) | Generates the hash key seed for a task.|
| size_t GenTaskKey(uint8_t inputKey[], size_t &keyLen, op::OpArgContext *args, const FVector<const aclTensor*> &inputs) const | Generates a key for task search.|
