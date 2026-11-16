# op\_dfx

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| ToString(const std::string &str) | Converts `std::string` to `ge::AscendString`.|
| OpDfxGuard(const char *file, int line, OpLevel level, const char*funcName, const char *paramNamesIn, const char*paramNamesOut, const INPUT_TUPLE &&in, const OUTPUT_TUPLE &&out) | Constructs an instance of the DFX statistics guard.|
| OpDfxGuard(const char *file, int line, OpLevel level, const char*funcName) | Constructs an instance of the DFX statistics guard.|
| OpDfxGuard(uint32_t id, const char *file, int line, OpLevel level, const char*funcName, const char *paramNames, const std::tuple<Args...> &t) | Constructs an instance of the DFX statistics guard.|
| OpDfxGuard(uint64_t id, DfxProfilingType type) | Constructs an instance of the DFX statistics guard.|
| OpDfxGuard() | Constructs an instance of the DFX statistics guard.|
| ValidDfxName([[maybe_unused]]char const *a, [[maybe_unused]]char const*b) | Checks if *a* equals *b*.|
| OpGetLogSequence() | Obtains the log sequence number.|
| GenSummaryItemId(const char *l2Name, const char*l0Name) | Generates a unique statistics ID using the L2 and L0 API names.|
| GenSummaryItemId(const char *l2Name, const char*l0Name, const char *opType) | Generates a unique statistics ID using the L2 and L0 API names and the operator type name.|
| GenKernelLauncherId(const char *l0Name) | Generates a kernel launch statistics ID using the L0 API name.|
| OpProfilingSwitch() | Constructs an instance of `OpProfilingSwitch`.|
| OpLogInfo() | Constructs an instance of `OpLogInfo`.|
| OpLogInfo(const OpLogInfo &rhs) | Constructs an instance of `OpLogInfo` by copying from another instance.|
| InitLevelZero() | Initializes L0 statistics.|
| InitLevelTwo() | Initializes L2 statistics.|
| PtrCastTo(From *ptr) | Converts `ptr` to a pointer of the specified type.|
| PtrCastTo(const From *ptr) | Converts `ptr` to a pointer of the specified type.|
| GenOpTypeId(const char *opName) | Generates a unique operator ID based on the given operator name.|
| IsDumpEnabled() | Checks if `dump` is enabled.|
| InitThreadLocalContext() | Initializes the local context of a thread.|
| AddInputTensorToThreadLocalCtx(const aclTensor *const t) | Adds an input tensor to the local context of a thread.|
| AddInputTensorToThreadLocalCtx(aclTensor *const t) | Adds an input tensor to the local context of a thread.|
| AddInputTensorToThreadLocalCtx([[maybe_unused]] T &t) | Adds an input tensor to the local context of a thread.|
| AddInputTensorsToThreadLocalCtx(const std::tuple<Args...> &t) | Adds multiple input tensors to the local context of a thread.|
| AddInputTensorToThreadLocalCtx(const aclTensorList *const t) | Adds multiple input tensors to the local context of a thread.|
| AddInputTensorToThreadLocalCtx(aclTensorList *const t) | Adds multiple input tensors to the local context of a thread.|
| AddOutputTensorToThreadLocalCtx(const aclTensor *const t) | Adds an output tensor to the local context of a thread.|
| AddOutputTensorToThreadLocalCtx(aclTensor *const t) | Adds an output tensor to the local context of a thread.|
| AddOutputTensorToThreadLocalCtx([[maybe_unused]] T &t) | Adds an output tensor to the local context of a thread.|
| AddOutputTensorsToThreadLocalCtx(const std::tuple<Args...> &t) | Adds multiple output tensors to the local context of a thread.|
| AddOutputTensorToThreadLocalCtx(const aclTensorList *const t) | Adds multiple output tensors to the local context of a thread.|
| AddOutputTensorToThreadLocalCtx(aclTensorList *const t) | Adds multiple output tensors to the local context of a thread.|
| CreateDfxProfiler(const char *funcName) | Creates a profiler with the given `funcName`.|
| CreateDfxProfiler(uint32_t id) | Creates a profiler with the given ID.|
| ToStr(const T &t, std::string &res, std::vector\<std::string\> &v, size_t &index) | Converts a given parameter to a string.|
| StringToVec(const char *paramNames, std::vector\<std::string\> &v) | Splits a given string by commas.|
| StringToVecWithBrackets(const char *paramNames, std::vector\<std::string\> &v) | Extracts the items inside strings such as `DFX_IN(aa, bb, cc)` and `DFX_OUT(aa, bb)`.|
| SplitStringAndPrint(std::string &res) | Splits a given long string into multiple shorter strings for printing.|
| BuildParamString(const char *paramNames, const std::tuple<Args...> &t) | Generates a parameter string using the given operator inputs.|
| BuildParamStringWithBrackets(const char *paramNames, const std::tuple<Args...> &t) | Generates a parameter string using the given operator inputs.|
| GenOpTypeId(const char *opName, const OP_RESOURCES &opResources) | Generates a unique operator ID using the given operator name and binary resource.|
| GenOpTypeId(const char *opName, const OP_SOC_RESOURCES &opResources) | Generates a unique operator ID using the given operator name and SoC binary resource.|
| GenInternalOpTypeId() | Generates an internal operator ID.|
| aclnnStatus CheckPhase1Params(aclOpExecutor **executor, uint64_t *workspaceSize) | Checks if the common parameters in the first phase are `nullptr`.|
