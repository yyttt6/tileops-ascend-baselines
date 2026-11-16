# CREATE\_EXECUTOR

## 宏功能

创建一个UniqueExecutor对象，该对象为aclOpExecutor的生成工厂类。

## 宏原型

```cpp
CREATE_EXECUTOR()
```

## 参数说明

无

## 约束说明

- 在使用aclOpExecutor前，必需要调用此宏创建。
- L2一阶段接口返回前，需要调用ReleaseTo将executor转移到L2接口入参的executor中。

## 调用示例

```cpp
// 创建executor
auto uniqueExecutor = CREATE_EXECUTOR();
```
