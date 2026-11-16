# AicpuTaskSpace Class

Manages the logic related to AI CPU task reuse, including creating and searching for a task.

The definition is as follows:

```cpp
class AicpuTaskSpace {
public:
        // Set the class of a dynamic-shape operator, either TensorFlow or CANN. The class-1 CANN operator is used by default.
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

For details about the members in this class, see the following table.

**Table 1** AicpuTaskSpace class members

| Attribute| Type| Default Value| Description|
| --- | --- | --- | --- |
| kHashSeed | const std::string | "" | Key value seed of the hash table that stores the map of a task.|
| opType_ | const ge::UnknowShapeOpType | 0 | Operator name.|
| unknownType_ | std::unique_ptr\<AicpuArgsHandler\> | null | Type of the operator.|
| isTf_ | std::unique_ptr\<AicpuExtInfoHandler\> | false | Whether a third-party framework or CANN framework is used.<br>  - `true`: A third-party operator framework is used. Currently, only the TensorFlow framework is supported.<br>  - `false`: The CANN framework is used.|
| hasInit_ | uint64_t | 0 | Whether a task is initialized.|
| inputRefIndexes_ | uint64_t | 0 | Whether the input is of the ref class.|
| outputRefIndexes_ | void* | null | Whether the output is of the ref class.|
| mutex_ | FVector<const aclTensor *> | null | Mutex of the task map.|
| hashMap_ | FVector<const aclTensor *> | null | Hash table that stores a task.|
