# aicpu\_task

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| PrintAicpuAllTimeStampInfo(const char *opType) | 打印AI CPU task执行时的系统时间戳。 |
| AppendTensor(aclOpExecutor *executor, const aclTensor*arg, V &l) | 将aclTensor指针塞入到容器中的模板函数。 |
| AppendTensor(aclOpExecutor *executor, const aclScalar*arg, V &l) | 将aclScalar指针塞入到容器中的模板函数。 |
| AppendTensor(aclOpExecutor *executor, const aclIntArray*arg, V &l) | 将aclIntArray指针塞入到容器中的模板函数。 |
| AppendTensor(aclOpExecutor *executor, const aclTensorList*arg, V &l) | 将aclTensorList指针塞入到容器中的模板函数。 |
| CreateTensorListImpl(aclOpExecutor *executor, OpArg &arg, TensorList &l) | 从输入的args中创建Tensor列表模板函数。 |
| CreateTensorList(aclOpExecutor *executor, OpArgList &t, TensorList &l) | 创建Tensor列表。 |
| Append1Byte(uint8_t *buf, uint8_t src) | 向指定的buffer后再拼接一个字节。 |
| AppendAttrForKey(const V &value, uint8_t *&key, size_t &keyLen) | 向task key字段中拼接属性信息的模板函数。 |
| AppendAttrForKey(const std::string &value, uint8_t *&key, size_t &keyLen) | 向task key字段中拼接const string &属性信息的函数。 |
| AppendAttrForKey(const std::string *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接const string指针属性信息的函数。 |
| AppendAttrForKey(std::string *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接string 指针属性信息的函数。 |
| AppendAttrForKey(const std::vector\<V\> &value, uint8_t *&key, size_t &keyLen) | 向task key字段中拼接vector属性信息的函数。 |
| AppendAttrForKey(const aclIntArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接const aclIntArray属性信息的函数。 |
| AppendAttrForKey(aclIntArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接aclIntArray属性信息的函数。 |
| AppendAttrForKey(const aclFloatArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接const aclFloatArray属性信息的函数。 |
| AppendAttrForKey(aclFloatArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接aclFloatArray属性信息的函数。 |
| AppendAttrForKey(const aclBoolArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接const aclBoolArray属性信息的函数。 |
| AppendAttrForKey(aclBoolArray *value, uint8_t*&key, size_t &keyLen) | 向task key字段中拼接aclBoolArray属性信息的函数。 |
| AddAicpuAttr(const aclIntArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(aclIntArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(const aclFloatArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(aclFloatArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(const aclBoolArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(aclBoolArray *value, const std::string &attrName, AicpuAttrs &attrs) | 添加AI CPU属性字段。 |
| AddAicpuAttr(const V &value, const std::string &attrName, AicpuAttrs &attrs) | 向AI CPU属性map中加入简单数据类型属性。 |
| GetTid() | 获取当前工作的线程id。 |
| aclnnAicpuFinalize() | AI CPU模块去初始化函数。 |
| CreatAicpuKernelLauncher(uint32_t opType, op::internal::AicpuTaskSpace &space, aclOpExecutor *executor, const FVector\<std::string\> &attrNames, op::OpArgContext*args) | 创建AI CPU task下发对象。 |
| AicpuTask() | AicpuTask类默认的构造函数。 |
| AicpuTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | AicpuTask类的构造函数(带入参)。 |
| AicpuTfTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | AI CPU tensorflow算子框架task构造函数。 |
| aclnnStatus Init(const FVector<const aclTensor *> &inputs, const FVector<aclTensor*> &outputs, const AicpuAttrs &attrs) | 算子框架task初始化函数。 |
| aclnnStatus Run(aclOpExecutor *executor, aclrtStream stream) | 算子框架task执行函数。 |
| AicpuTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | AI CPU task构造函数。 |
| AicpuCCTask(const std::string &opType, const ge::UnknowShapeOpType unknownType) | AI CPU CANN算子框架task构造函数。 |
| aclnnStatus SetIoTensors(aclOpExecutor *executor, op::OpArgContext*args) | 刷新task的输入、输出地址。 |
| void SetSpace(void *space) | 设置task所在的容器。 |
| void SetVisit(bool visit) | 设置task是否被占用。 |
| AicpuTaskSpace(const std::string &opType,  const ge::UnknowShapeOpType unknownType = ge::DEPEND_IN_SHAPE,  const bool isTf = false) | AI CPU task容器构造函数。 |
| AicpuTask *FindTask(aclOpExecutor*executor, op::OpArgContext *args, const FVector<const aclTensor*> &inputs) | 查找是否可复用的task。 |
| AicpuTask *GetOrCreateTask(aclOpExecutor*executor, const FVector\<std::string\> &attrNames, op::OpArgContext *args) | 获取新建或者复用的task。 |
| void SetRef(const size_t index, const bool isInput = true) | 设置指定索引为引用类型输入。 |
| bool IsRef(const size_t index, const bool isInput = true) const | 获取指定索引是否为引用类型输入。 |
| uint64_t CalcHostInputDataSize(const FVector<const aclTensor *> &inputs, size_t alignBytes) const | 获取输入为host侧数据的总大小。 |
| uint64_t CalcDeviceCacheSize(const FVector<const aclTensor *> &inputs, std::unique_ptr\<AicpuTask\> &aicpuTask) const | 计算device侧预留的内存大小。 |
| void Clear() | 清理缓存的task。 |
| static size_t GenHashBinary(const uint8_t *addr, uint32_t len) | 获取task哈希表键值种子。 |
| size_t GenTaskKey(uint8_t inputKey[], size_t &keyLen, op::OpArgContext *args, const FVector<const aclTensor*> &inputs) const | 生成task查找的key。 |
