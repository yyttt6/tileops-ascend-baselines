# pool\_allocator

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API Definition| Description|
| --- | --- |
| MallocPtr(size_t size) | Allocates memory of `size` bytes from the memory pool.|
| FreePtr(void *block) | Deallocates the data block to the memory pool.|
| PoolAllocator() | Indicates the `PoolAllocator` constructor.|
| PoolAllocator(const PoolAllocator\<U\> &) | Indicates the `PoolAllocator` constructor for copying data.|
| allocate(size_t n) | Allocates `n` T-type memories from the memory pool.|
| deallocate(T *p, [[maybe_unused]] size_t n) | Deallocates `p` to the memory pool.|
| construct(_Up *\_\_p,_Args &&...__args) | Indicates the constructor for calling `_Up` with the given parameter `__args` and address `__p`.|
| destroy(_Up *__p) | Indicates the destructor for calling `__p`.|
