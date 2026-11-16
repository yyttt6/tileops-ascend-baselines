# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API| Description|
| --- | --- |
| aclOpExecutor() | Indicates the context structure, used to record the running information of all APIs on the host. Almost all operations of the APIs on the host use this data structure as the medium. This function is a constructor.|
| CreateView(const aclTensor *tensor, const op::Shape &shape, int64_t offset) | Creates a view tensor for an existing aclTensor. Two tensors share the device memory. You can specify the shape and offset of the view tensor.|
| CreateView(const aclTensor *tensor, const op::Shape &oriShape, const op::Shape &storageShape, const op::Strides &oriStride, int64_t offset) | Creates a view tensor for an existing aclTensor. Two tensors share the device memory. You can specify the originShape, storageShape, originStride, and offset of the view tensor.|
| UpdateTensorAddr(void *workspaceAddr, const size_t size) | Updates the address of each workspace after workspace addresses are allocated.|
| GetWorkspaceAddr() | Obtains the start pointer of a workspace.|
| GetWorkspaceSize() | Obtains the size of the required workspace.|
| GetLinearWorkspaceSize() | This API is deprecated and can be ignored.|
| GetWorkspaceOffsets() | Obtains the workspace offsets.|
| SetWorkspaceOffsets(const op::FVector<uint64_t> &workspaceOffsets) | Sets the workspace offsets to be recorded.|
| Run() | Runs `aclOpExecutor` to execute tasks in the queue.|
| GetStream() | Obtains the current stream.|
| GetInputTensors() | Obtains the input aclTensor of the API on the host.|
| GetOutputTensors() | Obtains the output aclTensor of the API on the host.|
| GetLogInfo() | Obtains the log information stored in `aclOpExecutor`.|
| SetLogInfo(const op::internal::OpLogInfo &logInfo) | Obtains the log information stored in `aclOpExecutor`.|
| GetOpConfigInfo() | Obtains the configuration related to operator running.|
| SetOpConfigInfo(const op::OpConfigInfo &opConfigInfo) | Sets the configuration related to operator running.|
| SetStream(aclrtStream stream) | Sets the current stream.|
| AddTensorRelation(const aclTensor \*tensorOut, const aclTensor \*tensorMiddle) | Records the address equivalence relationship between two aclTensors for aclnn cache.|
| UpdateStorageAddr() | Updates the aclTensor address in the `aclOpExecutor` reuse scenario.|
| SetRepeatable() | Attempts to set the current `aclOpExecutor` to the reusable state.|
| IsRepeatable() | Checks whether the current `aclOpExecutor` is reusable.|
| FinalizeCache() | Saves the final data of the aclnn cache.|
| RepeatRunWithCache(void *workspaceAddr, const aclrtStream stream) | Runs the aclnn cache to execute tasks in the `aclOpExecutor` reuse scenario.|
| CheckLauncherRepeatable() | Checks whether `aclOpExecutor` reuse is allowed for each task in the `aclOpExecutor` task list.|
| AddCache() | Saves the aclnn cache in `aclOpExecutor` to global management.|
| DeleteCache() | Deletes the aclnn cache.|
| GetOpExecCache() | Obtains the aclnn cache recorded in `aclOpExecutor`.|
| SetIOTensorList() | Sets the input/output aclTensor of the API on the host.|
| GetGraph() | Obtains the execution graph of the API on the host.|
| GetMagicNumber() | Obtains the magic number to distinguish different objects.|
| UniqueExecutor(const char *funcName) | Indicates the constructor factory of `aclOpExecutor`.|
| UniqueExecutor() | Works as the constructor of `UniqueExecutor`.|
| get() | Obtains the `aclOpExecutor` pointer in `UniqueExecutor`.|
| ReleaseTo(aclOpExecutor **executor) | Passes the `aclOpExecutor` pointer in `UniqueExecutor` to the destination `aclOpExecutor` pointer.|
| UniqueExecutor(const UniqueExecutor &) | This API is deprecated and can be ignored.|
| GetOpExecCacheFromExecutor(aclOpExecutor *) | Converts an external `aclOpExecutor` to an aclnn cache object.|
| InitL2Phase1Context(const char *l2Name, [[maybe_unused]] aclOpExecutor **executor) | Initializes some DFX variable values in the first-phase API on the host.|
| InitL2Phase2Context([[maybe_unused]] const char\* l2Name, aclOpExecutor\* executor) | Initializes some DFX variable values in the second-phase API on the host.|
| InitL0Context(const char \*profilingName, aclOpExecutor* executor) | Initializes some DFX variable values of the L0 API.|
| CreatAiCoreKernelLauncher([[maybe_unused]] const char \*l0Name, uint32_t opType, aclOpExecutor \*executor, op::OpArgContext \*args) | Creates an AI Core task object.|
| CreatDSAKernelLauncher([[maybe_unused]] const char \*l0Name, uint32_t opType, DSA_TASK_TYPE dsaTask, aclOpExecutor \*executor, op::OpArgContext \*args) | Creates a DSA task object.|
| InferShape(uint32_t optype, op::OpArgList &inputs, op::OpArgList &outputs, op::OpArgList &attrs) | Obtains the shape inference result of a specified operator.|
| ConvertToTensor(const aclIntArray *value, op::DataType dataType) | Converts host data of the aclIntArray type into a host aclTensor.|
| ConvertToTensor(const aclBoolArray *value, op::DataType dataType) | Converts host data of the aclBoolArray type into a host aclTensor.|
| ConvertToTensor(const aclFloatArray *value, op::DataType dataType) | Converts host data of the aclFloatArray type into a host aclTensor.|
| ConvertToTensor(const aclFp16Array *value, op::DataType dataType) | Converts host data of the aclFp16Array type into a host aclTensor.|
| ConvertToTensor(const aclBf16Array *value, op::DataType dataType) | Converts host data of the aclBf16Array type into a host aclTensor.|
| ConvertToTensor(const T *value, uint64_t size, op::DataType dataType) | Converts host data of the T type into a host aclTensor.|
| ConvertToTensor(const aclScalar *value, op::DataType dataType) | Converts host data of the aclScalar type into a host aclTensor.|
| AddToKernelLauncherList(op::KernelLauncher *obj) | Adds an AI Core task to an execution queue.|
| AddToKernelLauncherListDvpp(uint32_t opType, op::KernelLauncher \*obj, op::OpArgContext \*args) | Adds a DVPP task to an execution queue.|
| AddToKernelLauncherListCopyTask(uint32_t opType, op::KernelLauncher *obj, op::OpArgList &inputs, op::OpArgList &outputs, op::OpArgList &workspace) | Adds a data copy task to an execution queue.|
| AddToKernelLauncherListAiCpu(int32_t opType, op::KernelLauncher \*obj, op::OpArgContext \*args) | Adds an AI CPU task to an execution queue.|
| CommonOpExecutorRun(void \*workspace, uint64_t workspaceSize, aclOpExecutor \*executor, aclrtStream stream) | Runs all tasks in the context based on the external `aclOpExecutor`, `workspace`, and `stream`.|
