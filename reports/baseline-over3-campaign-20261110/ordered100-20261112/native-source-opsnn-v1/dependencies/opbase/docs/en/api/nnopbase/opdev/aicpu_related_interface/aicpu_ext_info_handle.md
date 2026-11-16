# aicpu\_ext\_info\_handle

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| AicpuExtInfoHandler(const std::string &nodeName, const uint32_t inputNum, const uint32_t outputNum, const ge::UnknowShapeOpType unknownType) | Constructs an instance of the class that manages AI CPU extended parameters.|
| nodeName_(nodeName) | Operator name.|
| inputNum_(inputNum) | Number of operator inputs.|
| outputNum_(outputNum) | Number of operator outputs.|
| unknownType_(unknownType) | Operator type.|
| AicpuExtInfoHandler() | Constructs an instance of the class that manages AI CPU extended parameters.|
| GenTfExtBuffer(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, std::string &taskExtInfo) | Generates extended parameters for the TensorFlow operator framework.|
| GenCCExtBuffer( const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, std::string &taskExtInfo) | Generates extended parameters for the CANN operator framework.|
| Parse(const std::string &extInfo, uint8_t *hostAddr) | Parses extended parameters.|
| UpdateInputShape(const uint32_t inputIndex, const gert::Shape &inputShape) | Updates the input shape in an extended parameter.|
| UpdateOutputShape(const uint32_t outputIndex, const gert::Shape &outputShape) | Updates the output shape in an extended parameter.|
| GetOutputShapeAndType(const uint32_t outputIndex, gert::Shape &shape, ge::DataType &dataType) | Obtains shapes and data types of an extended parameter.|
| CopyH2D(const aclrtStream stream, const aclOpExecutor *executor, const uint64_t deviceExtMemSize, uint64_t &deviceCacheOffset) | Copies extended parameters from the host side to the device side.|
| CopyOutputShapeD2H() | Copies the updated output shape of an operator back from the device side to the host side.|
| GetExtInfoDeviceBuffer(const aclOpExecutor *executor, const uint64_t deviceExtMemSize, uint64_t &deviceCacheOffset) | Obtains the device-side reserved memory address for extended parameters.|
| GenerateKernelId() | Generates a kernel ID.|
| UpdateOutputShapeFromExtInfo(const FVector<aclTensor *> &outputs, aclrtStream stream) | Updates output shapes from the device side to the host side.|
| UpdateInputAndOutputShape(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, aclrtStream stream, const aclOpExecutor *executor, const uint64_t deviceExtMemSize, uint64_t &deviceCacheOffset) | Updates the actual input and output shapes from the host side to the device side before the second-phase execution.|
| SetSpace(void *space) | Sets the task space to be used.|
| ParseExtShapeType(AicpuExtInfo &aicpuExtInfo) | Parses shapes and data types in an extended parameter.|
| ParseExtInputShape(AicpuExtInfo &aicpuExtInfo) | Parses the input shape in an extended parameter.|
| ParseExtOutputShape(AicpuExtInfo &aicpuExtInfo) | Parses the output shape in an extended parameter.|
| AppendExtOpName(std::string &taskExtInfo) | Appends an operator name to an extended parameter.|
| AppendExtShapeType(std::string &taskExtInfo) | Appends shapes and data types to an extended parameter.|
| AppendExtBitMap(std::string &taskExtInfo) | Appends a bitmap to an extended parameter.|
| AppendExtInfoShape(const FVector<const aclTensor *> &tensors, const aicpu::FWKAdapter::FWKTaskExtInfoType type, std::string &taskExtInfo, bool isTf = false) | Appends shapes to an extended parameter.|
| AppendSessionInfo(std::string &taskExtInfo) | Appends a session to an extended parameter.|
| UpdateShape(const gert::Shape &shape, AicpuShapeAndType *const shapeAndType) | Updates a shape.|
| GetShapeAndType(const AicpuShapeAndType &shapeAndType, gert::Shape &shape, ge::DataType &dataType) | Obtains shapes and data types of an extended parameter.|
