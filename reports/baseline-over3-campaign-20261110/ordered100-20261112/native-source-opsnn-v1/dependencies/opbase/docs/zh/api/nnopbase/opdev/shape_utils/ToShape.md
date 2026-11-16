# ToShape

## 功能说明

将指定内存维度数据转为op::Shape（即gert::Shape）对象。

## 函数原型

- **用给定的dims维度数组构造op::Shape**

    ```cpp
    void ToShape(const int64_t *dims, uint64_t dimNum, op::Shape &shape)
    ```

- **用给定的op::ShapeVector维度数组构造op::Shape**

    ```cpp
    void ToShape(const op::ShapeVector &shapeVector, op::Shape &shape)
    ```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dims | 输入 | 源数据指针，指向存放每一维shape大小的内存。 |
| dimNum | 输入 | dims指向的源数据中，共有多少维shape信息。 |
| shapeVector | 输入 | 源数据，列表中存放每一维的shape大小。 |
| shape | 输出 | 通过源数据构造出的Shape对象。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 生成一个shape信息为[1, 2, 3, 4, 5]的Shape对象。
void Func() {
    int64_t myArray[5] = {1, 2, 3, 4, 5};
    Shape shape;
    ToShape(myArray, 5, shape);
}

```
