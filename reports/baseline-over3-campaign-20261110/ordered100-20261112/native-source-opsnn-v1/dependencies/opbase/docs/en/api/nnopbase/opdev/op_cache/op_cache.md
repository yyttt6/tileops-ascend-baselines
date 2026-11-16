# op\_cache

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API Definition| Description|
| --- | --- |
| GetCacheBuf() | Obtains the buffer required for the aclnn cache.|
| CheckCacheable() | Checks whether an aclnn API on the host supports the aclnn cache.|
| AddSeperator() | Adds a separator to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const T *addr, uint64_t size) | Adds key information of the const T type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclTensor *tensor) | Adds key information of the const aclTensor type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(aclTensor *tensor) | Adds key information of the aclTensor type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclScalar *scalar) | Adds key information of the const aclScalar type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclIntArray *value) | Adds key information of the const aclIntArray type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclBoolArray *value) | Adds key information of the const aclBoolArray type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclFloatArray *value) | Adds key information of the const aclFloatArray type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclFp16Array *value) | Adds key information of the const aclFp16Array type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclTensorList *tensors) | Adds key information of the const aclTensorList type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclScalarList *scalars) | Adds key information of the const aclScalarList type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(aclTensorList *tensors) | Adds key information of the aclTensorList type to the hashkey calculation basis of the aclnn cache.|
| AddParamToBuf(aclScalarList *scalars) | Adds key information of the aclScalarList type to the hashkey calculation basis of the aclnn cache.|
| AddParamToBuf(const std::string &s) | Adds key information of the const string & type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const aclDataType dtype) | Adds key information of the const aclDataType type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const char *c) | Adds key information of the const char type to the hashkey computation of the aclnn cache.|
| AddParamToBuf(char *c) | Adds key information of the char type to the hashkey computation of the aclnn cache.|
| AddParamToBuf() | Works as the constructor to add key information to the hashkey computation of the aclnn cache.|
| AddParamToBuf(const T &value) | Adds key information of the const T & type to the hashkey computation of the aclnn cache.|
| AddOpConfigInfoToBuf() | Adds the operator configuration, for example, deterministic computation, to the hashkey computation of the aclnn cache.|
| InitExecutorCacheThreadLocal() | Initializes thread variables related to the aclnn cache.|
| GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize) | Obtains the matched aclnn cache.|
| CalculateHashKey(const std::tuple<Args...> &t) | Computes the hashkey of the aclnn cache based on the input and output.|
| GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize, const char*api, const INPUT_TUPLE &in, const OUTPUT_TUPLE &out) | Attempts to obtain the aclnn cache that matches the specified input and output.|
| OpCacheKey() | Works as the constructor of `OpCacheKey`, which is the key data structure of the aclnn cache.|
| OpCacheKey(uint8_t *buf\_, size_t len_) | Works as the constructor of `OpCacheKey`, which is the key data structure of the aclnn cache.|
| OpCacheKey(const OpCacheKey &key) | Works as the constructor of `OpCacheKey`, which is the key data structure of the aclnn cache.|
| ToString() | Prints `OpCacheKey` as a string.|
| operator() | Works as the equivalence judgment function of `OpCacheKey`.|
| SetOpCacheKey(OpCacheKey &key) | Sets `OpCacheKey` corresponding to the current input or output.|
| AddrRule() | Describes the mappings between the aclTensors recorded in the aclnn cache and the external input parameters of the APIs on the host and updates the address of the aclnn cache. This function is a constructor.|
| AddrRule(bool w, uint64_t offset, int inx, int64_t l2Offset) | Describes the mappings between the aclTensors recorded in the aclnn cache and the external input parameters of the APIs on the host and updates the address of the aclnn cache. This function is a constructor.|
| OpExecCache() | Works as the constructor of `OpExecCache`, which describes aclnn cache information.|
| InitOpCacheKey() | Initializes the hashkey of the aclnn cache.|
| InitCacheData() | Initializes the external information required by the aclnn cache.|
| RecordAddrRule(const aclTensor *t, AddrRule &rule) | Adds an object that records the mappings between the aclTensor and the external input parameters of the APIs on the host during aclnn cache generation.|
| AddLaunchTensor(const aclTensor *t, size_t dataLen) | Adds a cache record for an aclTensor during aclnn cache generation.|
| AddLaunchData(size_t dataLen) | Requests a space of a specified size in the memory where the aclnn cache is located to fill in the cache content.|
| SetCacheBuf(void *buf) | Sets the temporary buffer used by the aclnn cache.|
| UpdateTensorAddr(void *workspaceAddr, const std::vector<void*> &tensors) | Updates the actual input address of the API on the host to the cache during aclnn cache use.|
| SetBlockDim(uint32_t blockDim) | Sets the number of cores used by the cache object during aclnn cache generation. This API will be deprecated. Please use SetNumBlocks(uint32_t numBlocks).|
| SetNumBlocks(uint32_t numBlocks) | Sets the number of cores used by the cache object during aclnn cache generation.|
| SetCacheTensorInfo(void *infoLists) | Sets the aclTensor attribute information to enable the aclnn cache to support profiling reporting.|
| GetCacheTensorInfo(int index) | Obtains the aclTensor attribute information to enable the aclnn cache to support profiling reporting.|
| AddTensorRelation(const aclTensor *tensorOut, const aclTensor*tensorMiddle) | Adds the memory equivalence relationship between two aclTensors to the aclnn cache.|
| NewLaunchCache(size_t *offset, size_t*cap, R runner) | Generates a new cache.|
| OldCacheClear() | Clears an old cache.|
| Finalize() | Finishes the aclnn cache generation and sets the available flag bit.|
| SetWorkspaceSize(uint64_t val) | Sets the size of the workspace used by the cache object during aclnn cache generation.|
| GetWorkspaceSize() | Obtains the size of the workspace used by the cache object during aclnn cache use.|
| GetHash() | Obtains the hashkey of the current cache.|
| GetOpCacheKey() | Obtains the hashkey of the current cache.|
| MarkOpCacheInvalid() | Sets the current aclnn cache to invalid state.|
| IsOpCacheValid() | Queries whether the current aclnn cache is in invalid state.|
| DoSummaryProfiling(int index) | Reports the profiling summary during aclnn cache use.|
| RestoreThreadLocal(int index) | Restores the current thread context during aclnn cache use.|
| Run(void *workspaceAddr, const aclrtStream stream, const std::vector<void*> &tensors) | Runs a host API through the aclnn cache.|
| CanUse() | Sets the availability status of the aclnn cache.|
| SetUse() | Obtains the availability status of the aclnn cache.|
| GetShrinkList() | Obtains the current shrink list of the aclnn cache.|
| GetStorageRelation() | Obtains the aclTensor memory equivalence table recorded in the aclnn cache.|
| OpCacheValue() | Works as the constructor of `OpCacheValue`, which is the data node encapsulation of the aclnn cache in the least recently used (LRU).|
| OpCacheValue(OpExecCache *cache, OpCacheKey &key) | Works as the constructor of `OpCacheValue`, which is the data node encapsulation of the aclnn cache in the least recently used (LRU).|
| OpCacheValue(const OpCacheValue &value) | Works as the constructor of `OpCacheValue`, which is the data node encapsulation of the aclnn cache in the least recently used (LRU).|
| OpCacheValue(OpCacheValue &&value) | Works as the constructor of `OpCacheValue`, which is the data node encapsulation of the aclnn cache in the least recently used (LRU).|
| GetOpExecCache(uint64_t hash) | Obtains an aclnn cache by hashkey.|
| GetOpExecCache(OpCacheKey &key) | Obtains an aclnn cache by hashkey.|
| AddOpExecCache(OpExecCache *exec) | Adds an aclnn cache object to the global management of the aclnn cache.|
| RemoveExecCache(OpExecCache *exec) | Removes an aclnn cache object to the global management of the aclnn cache.|
