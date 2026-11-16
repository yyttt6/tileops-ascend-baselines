# pool\_allocator

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| MallocPtr(size_t size) | 从内存池中申请size字节的内存。 |
| FreePtr(void *block) | 将data block释放回内存池。 |
| PoolAllocator() | PoolAllocator构造函数。 |
| PoolAllocator(const PoolAllocator\<U\> &) | PoolAllocator拷贝构造函数。 |
| allocate(size_t n) | 从内存池中申请n个T类型大小的内存。 |
| deallocate(T *p, [[maybe_unused]] size_t n) | 将p释放回内存池。 |
| construct(_Up *\_\_p,_Args &&...__args) | 用给定的参数\_\_args和地址\_\_p调用_Up的构造函数。 |
| destroy(_Up *__p) | 调用__p的析构函数。 |
