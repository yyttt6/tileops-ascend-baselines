# AicpuTask类

AicpuTask类用于管理AI CPU task相关的一些参数和方法，包括组装task、task初始化、task下发、task执行等功能。

具体定义如下：

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
    // 长度可能不够，后续考虑长度可扩展
    uint8_t inputKey_[kAicpuKeyBufLen] = {};
    size_t keyLen_ = 0;
    bool isVisit_ = false;
    uint64_t deviceExtMemSize_ = 0;
    uint64_t deviceCacheOffset_ = 0;
};
```

类成员属性的详细介绍请参考下表。

**表 1**  AicpuTask类成员说明

| 属性名 | 属性类型 | 默认值 | 属性说明 |
| --- | --- | --- | --- |
| opType_ | const std::string | "" | Task对应的AI CPU算子名。 |
| unknownType_ | const ge::UnknowShapeOpType | 0 | 标识算子shape是基类，具体类型UnknowShapeOpType。 |
| argsHandle_ | std::unique_ptr\<AicpuArgsHandler\> | null | 管理task封装参数的对象。 |
| extInfoHandle_ | std::unique_ptr\<AicpuExtInfoHandler\> | null | 管理task封装拓展参数的对象。 |
| launchId_ | uint64_t | 0 | 用于profiling采集的任务下发id。 |
| summaryItemId_ | uint64_t | 0 | 用于统计条目的id。 |
| space_ | void* | null | 记录task属于哪个task管理集合。 |
| inputs_ | FVector<const aclTensor *> | null | task对应算子的输入tensor指针列表。 |
| outputs_ | FVector<const aclTensor *> | null | task对应算子的输出tensor指针列表。 |
| inputKey_ | uint8_t [kAicpuKeyBufLen] | 0 | task对应的key字段。 |
| keyLen_ | size_t | 0 | key字段长度。 |
| isVisit_ | bool | false | 当前task是否正被使用。 |
| deviceExtMemSize_ | uint64_t | 0 | device侧预留的内存。 |
| deviceCacheOffset_ | uint64_t | 0 | device侧预留内存的偏移。 |
