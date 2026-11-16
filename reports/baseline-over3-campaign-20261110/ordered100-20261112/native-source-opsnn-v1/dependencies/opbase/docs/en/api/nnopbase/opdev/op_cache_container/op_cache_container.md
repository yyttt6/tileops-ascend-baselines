# op\_cache\_container

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| ListHead() | Initializes a doubly linked list.|
| Add(ListHead *head) | Adds `head` at the beginning of the current linked list.|
| Del() | Deletes the current node from the linked list.|
| Empty() | Checks if a linked list is empty.|
| HlistNode() | Initializes a hash table node.|
| HlistHead() | Initializes the head of a linked list in a hash table.|
| Add(HlistNode *node) | Adds `node` at the beginning of a linked list in a hash table.|
| Lru() | Constructs a Least Recently Used (LRU) linked list.|
| Head() | Obtains the head of a Least Recently Used (LRU) linked list.|
| Tail() | Obtains the tail of a Least Recently Used (LRU) linked list.|
| Sentinel() | Obtains the sentinel node of a Least Recently Used (LRU) linked list.|
| Active(ListHead &entry) | Moves `entry` to the head of a Least Recently Used (LRU) linked list.|
| Del(ListHead &entry) | Deletes `entry` from a Least Recently Used (LRU) linked list.|
| OpCacheContainerIterator(pointer ptr, ListHead *sentinel, bool reverse = false) | Constructs `OpCacheContainerIterator`.|
| OpCacheContainerIterator(const OpCacheContainerIterator<KeyType, ValueType> &iter) | Constructs an instance of `OpCacheContainerIterator` by copying from another instance.|
| OpCacheContainer(const hasher &hash = hasher(), const key_equal &equal = key_equal()) | Constructs in instance of `OpCacheContainer` (a hash function is required).|
| OpCacheContainer() | Constructs an instance of `OpCacheContainer`.|
| begin() | Obtains the head node of `OpCacheContainer`.|
| init(size_t capacity) | Initializes `OpCacheContainer`.|
| find(const key_type &key) | Searches for a value in `OpCacheContainer` using the given key.|
| insert(reference value) | Inserts a value into `OpCacheContainer`.|
| erase(reference value) | Deletes a value from `OpCacheContainer`.|
| rbegin() | Obtains the reverse head node of `OpCacheContainer`.|
| rend() | Obtains the reverse tail node of `OpCacheContainer`.|
| size() | Obtains the size of `OpCacheContainer`.|
| bucket(const KeyType &key) | Obtains the bucket of `OpCacheContainer` using the given key.|
| bucket_count() | Obtains the number of buckets in `OpCacheContainer`.|
| value_type() | Obtains the value constructor of `OpCacheContainer`.|
| GetBucket(const key_type &key) | Obtains the bucket of `OpCacheContainer` using the given key.|
