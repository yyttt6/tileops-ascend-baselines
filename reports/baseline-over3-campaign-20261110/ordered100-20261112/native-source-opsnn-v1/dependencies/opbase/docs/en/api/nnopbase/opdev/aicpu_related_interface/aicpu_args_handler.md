# aicpu\_args\_handler

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API Definition| Description|
| --- | --- |
| AicpuArgsHandler(const std::string &opType, const std::string &nodeName, const uint32_t ioNum, const bool needDeviceExt) | Indicates the auxiliary class used by the AI CPU framework to construct AI CPU opTask.|
| opType_(opType) | Indicates the operator name.|
| nodeName_(nodeName) | Indicates the node name, which is the same as the operator name. This is a reserved parameter.|
| ioNum_(ioNum) | Indicates the total number of inputs and outputs of an operator.|
| needDeviceExt_(needDeviceExt) | Indicates whether additional device memory needs to be allocated.|
| args_({}) | Indicates the object of the parameters for interacting with Runtime. It is used to deliver computing tasks.|
| GetIoAddr() | Obtains the pointer to memory for storing the input/output address in the args memory.|
| get() | Obtains the start address of memory managed by the smart pointer.|
| GetExtInfoAddr() | Obtains the pointer to memory for storing extended parameters in args.|
| GetArgs() | Obtains the pointer to the args memory.|
| GetArgsEx() | Obtains the pointer to the object for interacting with Runtime.|
| GetNodeName() | Obtains the node name, which is the same as the operator name.|
| GetIoNum() | Obtains the total number of inputs and outputs of an operator.|
| GetHostInputSize() | Obtains the input size of host memory.|
| GetInputAddrAlignBytes() | Obtains the byte alignment of specified address allocation.|
| SetSpace(void *space) | Sets the operator space to which the args handler belongs.|
| MallocMem() | Allocates memory for args.|
| ResetHostInputInfo() | Resets the input information on the host.|
| AddHostInput(const size_t idx, void *data, const size_t srcSize, const size_t alignSize) | Records the input information on the host.|
| UpdateIoAddr(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, const aclrtStream stream, aclOpExecutor *executor, const uint64_t deviceExtMemSize, const uint64_t deviceCacheOffset) | Updates the input and output addresses in args.|
| AicpuArgsHandler() | Indicates the auxiliary class used by the AI CPU framework to construct AI CPU opTask.|
| UpdateDeviceExtInfoAddr(void *deviceExtInfoAddr) | Updates the address for storing extended parameters on the device.|
| SetLaunchArgs(const size_t argSize) | Sets the sizes of parameters in args.|
| GetDeviceCacheAddr(void *&deviceAddr, aclOpExecutor*executor, const uint64_t deviceCacheOffset) | Obtains the address of reserved memory on the device.|
| AicpuCCArgsHandler(const std::string &opType, const std::string &nodeName, const uint32_t ioNum, const bool needDeviceExt) | Indicates the class for managing AI CPU CANN framework parameters.|
| AicpuArgsHandler(opType, nodeName, ioNum, needDeviceExt) | Indicates the auxiliary class used by the AI CPU framework to construct AI CPU opTask.|
| GenCCArgs(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, const AicpuAttrs &attrs, std::string &taskInfo) | Generates the nodedef and head information in args of the CANN operator.|
| BuildCCArgs(const std::string &argData, const std::string &kernelName, const std::string &soName, const size_t extInfoSize) | Encapsulates parameters in args of the CANN operator.|
| SetHostArgs(const std::string &argData, const size_t extInfoSize) | Sets the address and length of memory in args.|
| SetOffsetArgs() | Sets the offset value in args.|
| AicpuTfArgsHandler(const std::string &opType, const std::string &nodeName, const uint32_t ioNum, const bool needDeviceExt) | Indicates the class for managing AI CPU TensorFlow framework parameters.|
| GenTfArgs(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, const AicpuAttrs &attrs, STR_FWK_OP_KERNEL &fwkOpKernel, std::string &taskInfo) | Generates the nodedef and head information in args of the CANN operator.|
| BuildTfArgs(STR_FWK_OP_KERNEL &fwkOpKernel, const std::string &taskInfo, const size_t extInfoSize) | Encapsulates parameters in args of the TensorFlow operator.|
| GenNodeDef(const FVector<const aclTensor *> &inputs, const AicpuAttrs &attrs, ge::Buffer &nodeDefBytes) | Generates the operator description, including the input and output addresses and shapes.|
| UpdateKernelId() | Updates the Kernel field in args to match the kernel cache on the device.|
