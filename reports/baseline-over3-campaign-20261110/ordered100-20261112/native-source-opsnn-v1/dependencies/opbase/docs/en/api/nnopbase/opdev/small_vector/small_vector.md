# small\_vector

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| PtrToValue(const void *const ptr) | Converts `void*` to `uint64_t data`.|
| ValueToPtr(const uint64_t value) | Converts `uint64_t data` to `void*`.|
| VPtrToValue(const std::vector<void *> v_ptr) | Converts `void*` elements in a `vector` container to `uint64_t` elements.|
| PtrToPtr(TI *const ptr) | Converts a pointer variable of any type (the pointer address cannot be changed) to a pointer variable of another type.|
| PtrToPtr(const TI *const ptr) | Converts a pointer variable of any type (the pointer address and the value it points to cannot be changed) to a pointer variable of another type (the value it points to cannot be changed).|
| PtrAdd(T *const ptr, const size_t max_buf_len, const size_t idx) | Returns `ptr + idx` if `ptr` is not `null` and `idx` < `max_buf_len`; otherwise, returns `null`.|
| SmallVector(const allocator_type &alloc = Alloc()) | Constructs a default instance of `SmallVector`.|
| SmallVector(const size_type count, const T &value, const allocator_type &alloc = Alloc()) | Constructs a `SmallVector` instance and initializes `count` elements to the same value.|
| SmallVector(const size_type count, const allocator_type &alloc = Alloc()) | Constructs a `SmallVector` instance with `count` initial elements.|
| SmallVector(InputIt first, const InputIt last, const allocator_type &alloc = Alloc()) | Constructs a `SmallVector` instance and initializes elements using an iterator range.|
| SmallVector(const SmallVector &other) | Constructs an instance of `SmallVector` by copying from another instance. It copies elements from `other` and storing them in newly allocated memory.|
| SmallVector(SmallVector &&other) | Constructs an instance of `SmallVector` by moving resources of an existing vector into a newly created vector.|
| SmallVector(const std::initializer_list\<T\> init, const allocator_type &alloc = Alloc()) | Constructs an instance of `SmallVector`.|
| clear() | Clears `SmallVector`.|
| ClearElements() | Clears `SmallVector` and returns the initial address of the iterator.|
| FreeStorage() | Clears memory.|
| at(const size_type index) | Returns a reference to the element at the specified `index` in `SmallVector`.|
| front() | Returns a reference to the first element in `SmallVector`.|
| begin() | Returns an iterator pointing to the first element in the `SmallVector` container.|
| back() | Returns a reference to the last element in `SmallVector`.|
| rbegin() | Returns a reverse iterator pointing to the last element in the `SmallVector` container.|
| data() | Returns a pointer to the first element.|
| GetPointer() | Returns a pointer to the first element.|
| cbegin() | Returns a `const_iterator` pointing to the first element in the `SmallVector` container.|
| cend() | Returns a `const_iterator` pointing to the element following the last element in the `SmallVector` container.|
| crbegin() | Returns a `const_iterator` pointing to the last element in the `SmallVector` container.|
| rend() | Returns a reverse iterator pointing to the element before the first element in the `SmallVector` container, which is considered its reverse end.|
| crend() | Returns a `const` reverse iterator pointing to the element before the first element in the `SmallVector` container, which is considered its reverse end.|
| size() | Returns the number of elements in the `SmallVector` container.|
| reserve(const size_type new_cap) | Increases the capacity of the `SmallVector` container when `new_cap` exceeds the allocated capacity so that the storage space is sufficient to hold `new_cap` elements.|
| capacity() | Returns the size of the allocated storage capacity.|
| insert(const_iterator const pos, const T &value) | Inserts an element before the specified position in the `SmallVector` container.|
| insert(const_iterator const pos, T &&value) | Inserts a `value` element before the specified position in the `SmallVector` container. The `value` is inserted by shallow copy.|
| insert(const_iterator const pos, const size_type count, const T &value) | Inserts `count` copies of a `value` element before the specified position in the `SmallVector` container.|
| insert(const_iterator const pos, const InputIt first, const InputIt last) | Copies elements from the specified range `[first, last)` and inserts them before the specified position in the `SmallVector` container.|
| insert(const_iterator const pos, const std::initializer_list\<T\> value_list) | Inserts all elements of an array of type `T` before the specified position in the `SmallVector` container.|
| emplace(const_iterator const pos, Args &&...args) | Inserts a new element before `pos` in the `SmallVector` container. The element is constructed using `args`.|
| erase(const_iterator const pos) | Deletes an element at the specified position in the `SmallVector` container.|
| erase(const_iterator const first, const_iterator const last) | Deletes elements in the range `[first, last)` from the `SmallVector` container.|
| push_back(const T &value) | Adds an element to the end of the `SmallVector` container.|
| push_back(T &&value) | Adds a `value` element to the end of the `SmallVector` container. The `value` is added by shallow copy.|
| emplace_back(Args &&...args) | Inserts an element at the end of the `SmallVector` container. The element is constructed using `args`.|
| pop_back() | Deletes the last element from the `SmallVector` container.|
| resize(const size_type count) | Resizes the `SmallVector` container to `count` elements. If `count` is less than the current container size, the first `count` elements are retained. Otherwise, the corresponding number of elements are added to the end of the container using the default constructor.|
| resize(const size_type count, const T &value) | Resizes the `SmallVector` container to `count` elements. If `count` is less than the current container size, the first `count` elements are retained. Otherwise, `value` is copied into the newly added elements.|
| swap(SmallVector &other) | Swaps the contents of the `SmallVector` container with another `SmallVector` container.|
| GetPointer(const size_type idx = 0UL) | Returns the address of the specified position.|
| InitStorage(const size_type size) | Initializes the entire container memory. If `size` is less than the current capacity, the memory of the current capacity is initialized. If `size` is greater than the current capacity, the container's memory is resized to the specified `size` and then initialized.|
| CopyRange(T *iter, InputIt first, const InputIt last) | Copies the memory within the range `[first, last)` to the position specified by the iterator `iter`.|
| MoveFrom(SmallVector &other) | Copies memory from the `other` container to the current container.|
| CheckOutOfRange(const size_type index) | Checks whether `index` is out of the container range.|
| ExpandCap(const size_type range_begin, const size_type range_len) | Allocates a new memory block whose size is the original container capacity plus `range_len`. If the new capacity is less than twice the original capacity, it is expanded to twice the original capacity. The memory is expanded by `range_len` at the position of `range_begin` in the original `SmallVector` container. Then, the contents are copied to the new memory. The original container memory will be released.|
| ExpandSize(const size_type range_begin, const size_type range_len) | Expands the memory by `range_len` at the position of `range_begin` in the `SmallVector` container.|
| Expand(const size_type range_begin, const size_type range_len) | Expands the memory of the `SmallVector` container. If the expanded memory is greater than the maximum capacity of the container, the expansion is performed in `ExpandCap` mode. Otherwise, the expansion is performed in `ExpandSize` mode.|
| Shrink(const size_type range_begin, const size_type range_end) | Deletes the memory from `range_begin` to `range_end` in the `SmartVector` container.|
| swap(op::internal::SmallVector<T, N, Alloc> &sv1, op::internal::SmallVector<T, N, Alloc> &sv2) | Swaps the elements of two `SmallVector` containers, `sv1` and `sv2`.|
