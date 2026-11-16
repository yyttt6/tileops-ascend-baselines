# AicpuTask Class

Manages parameters and methods related to AI CPU tasks, including task encapsulation, initialization, delivery, and execution.

The definition is as follows:

```cpp
class AicpuTask {
public:
    AicpuTask(const std::string &opType, const ge::UnknowShapeOpType unknownType)
        : opType_(opType), unknownType_(unknownType) {}
    virtual ~AicpuTask() = default;
    virtual aclnnStatus Init(const FVector<const aclTensor *> &inputs, const FVector<aclTensor *> &outputs,
                             const AicpuAttrs &attrs) = 0;
    virtual aclnnStatus Run(aclOpExecutor *executor, aclrtStream stream) = 0;
    aclnnStatus SetIoTensors(aclOpExecutor *executor, op::OpArgContext *args);
    friend class AicpuTaskSpace;
    void SetSpace(void *space)
    {
        space_ = space;
    }
    void SetVisit(bool visit);
protected:
    const std::string opType_;
    const ge::UnknowShapeOpType unknownType_;
    std::unique_ptr<AicpuArgsHandler> argsHandle_;
    std::unique_ptr<AicpuExtInfoHandler> extInfoHandle_;
    uint64_t launchId_ = 0U;
    uint64_t summaryItemId_ = 0U;
    void *space_ = nullptr;
    FVector<const aclTensor *> inputs_;
    FVector<aclTensor *> outputs_;
    // The length may be insufficient and can be extended in the future.
    uint8_t inputKey_[kAicpuKeyBufLen] = {};
    size_t keyLen_ = 0;
    bool isVisit_ = false;
    uint64_t deviceExtMemSize_ = 0;
    uint64_t deviceCacheOffset_ = 0;
};
```

For details about the members in this class, see the following table.

**Table 1** AicpuTask class members

| Attribute| Type| Default Value| Description|
| --- | --- | --- | --- |
| opType_ | const std::string | "" | Name of the AI CPU operator corresponding to a task.|
| UnknownType_ | const ge::UnknowShapeOpType | 0 | A flag indicating that the operator shape is a base class. The specific type is `UnknowShapeOpType`.|
| argsHandle_ | std::unique_ptr\<AicpuArgsHandler\> | null | Object for managing task encapsulation parameters.|
| extInfoHandle_ | std::unique_ptr\<AicpuExtInfoHandler\> | null | Object for managing extended task encapsulation parameters.|
| launchId_ | uint64_t | 0 | ID of a profiling task.|
| summaryItemId_ | uint64_t | 0 | ID of a summary item.|
| space_ | void* | null | Task management set to which a task belongs.|
| inputs_ | FVector<const aclTensor *> | null | List of input tensor pointers of the operator corresponding to a task.|
| outputs_ | FVector<const aclTensor *> | null | List of output tensor pointers of the operator corresponding to a task.|
| inputKey_ | uint8_t [kAicpuKeyBufLen] | 0 | Key field corresponding to a task.|
| keyLen_ | size_t | 0 | Length of the key field.|
| isVisit_ | bool | false | Whether the current task is in use.|
| deviceExtMemSize_ | uint64_t | 0 | Memory reserved on the device.|
| deviceCacheOffset_ | uint64_t | 0 | Offset of the reserved device memory.|
