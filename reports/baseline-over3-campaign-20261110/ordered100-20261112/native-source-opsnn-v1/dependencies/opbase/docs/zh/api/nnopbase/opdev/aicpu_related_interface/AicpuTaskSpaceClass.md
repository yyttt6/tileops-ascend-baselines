# AicpuTaskSpace类

AicpuTaskSpace类用于管理AI CPU task复用相关的逻辑，包括新建task、查找task等功能。

具体定义如下：

```cpp
class AicpuTaskSpace {
public:
    // 需要设置第几类动态shape算子，Tensorflow or CANN，默认设置为CANN第一类算子
    AicpuTaskSpace(const std::string &opType,
                   const ge::UnknowShapeOpType unknownType = ge::DEPEND_IN_SHAPE,
                   const bool isTf = false)
        : opType_(opType), unknownType_(unknownType), isTf_(isTf) {}
    AicpuTask *FindTask(aclOpExecutor *executor, op::OpArgContext *args,
                        const FVector<const aclTensor *> &inputs);
    AicpuTask *GetOrCreateTask(aclOpExecutor *executor, const FVector<std::string> &attrNames,
                               op::OpArgContext *args);
    void SetRef(const size_t index, const bool isInput = true);
    bool IsRef(const size_t index, const bool isInput = true) const;
    uint64_t CalcHostInputDataSize(const FVector<const aclTensor *> &inputs, size_t alignBytes) const;
    uint64_t CalcDeviceCacheSize(const FVector<const aclTensor *> &inputs,
                                 std::unique_ptr<AicpuTask> &aicpuTask) const;
    void Clear()
    {
        hashMap_.clear();
    }
    friend class AicpuTask;
private:
    static constexpr uint64_t kHashSeed = 0x9e3779b9U;
    static size_t GenHashBinary(const uint8_t *addr, uint32_t len);
    size_t GenTaskKey(uint8_t inputKey[], size_t &keyLen, op::OpArgContext *args,
                      const FVector<const aclTensor *> &inputs) const;
    const std::string opType_;
    const ge::UnknowShapeOpType unknownType_;
    const bool isTf_;
    bool hasInit_ = false;
    std::set<size_t> inputRefIndexes_;
    std::set<size_t> outputRefIndexes_;
    std::mutex mutex_;
    using HashMap = std::unordered_map<size_t, std::vector<std::unique_ptr<AicpuTask>>>;
    HashMap hashMap_;
};
```

类成员属性的详细介绍请参考下表。

**表 1**  AicpuTaskSpace类成员说明

| 属性名 | 属性类型 | 默认值 | 属性说明 |
| --- | --- | --- | --- |
| kHashSeed | const std::string | "" | 存储task的map的哈希表的键值种子。 |
| opType_ | const ge::UnknowShapeOpType | 0 | 算子名称。 |
| unknownType_ | std::unique_ptr\<AicpuArgsHandler\> | null | 标识是几类算子。 |
| isTf_ | std::unique_ptr\<AicpuExtInfoHandler\> | false | 标识是执行第三方算子框架还是CANN算子框架。<br>  - 取值true时：采用第三方算子框架，当前仅支持Tensorflow框架。<br>  - 取值false时：采用CANN算子框架。 |
| hasInit_ | uint64_t | 0 | 标识task是否被初始化。 |
| inputRefIndexes_ | uint64_t | 0 | 标识输入是否为ref类。 |
| outputRefIndexes_ | void* | null | 标识输出是否为ref类。 |
| mutex_ | FVector<const aclTensor *> | null | task map的锁。 |
| hashMap_ | FVector<const aclTensor *> | null | 存储task的hash表。 |
