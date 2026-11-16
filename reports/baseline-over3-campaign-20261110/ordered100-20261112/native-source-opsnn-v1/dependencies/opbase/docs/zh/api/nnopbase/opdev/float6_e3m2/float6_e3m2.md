# Float6E3M2

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| FromBits() | 返回用于从位模式构造Float6E3M2对象的标签。 |
| Float6E3M2() | struct Float6E3M2的默认构造函数，初始化为零值。 |
| Float6E3M2(uint8_t bits, FromBitsTag) | struct Float6E3M2的构造函数，从原始位模式构造对象。 |
| Float6E3M2(float v) | struct Float6E3M2的构造函数，从float值构造对象。 |
| operator float() | 将Float6E3M2转换为float。 |
| operator double() | 将Float6E3M2转换为double。 |
| Epsilon() | 返回Float6E3M2数据类型的机器epsilon值。 |
| Highest() | 返回Float6E3M2数据类型的最大有限值。 |
| Lowest() | 返回Float6E3M2数据类型的最小有限值。 |
| MinPositiveNormal() | 返回Float6E3M2数据类型的最小正规格化数。 |
| IsZero() | 判断当前Float6E3M2数值是否为零。 |
| IsNaN() | 判断当前Float6E3M2数值是否为NaN。 |
| IsInf() | 判断当前Float6E3M2数值是否为无穷大。 |
| isinf(const Float6E3M2 &a) | 判断给定的Float6E3M2数值a是否为无穷大。 |
| isnan(const Float6E3M2 &a) | 判断给定的Float6E3M2数值a是否为NaN。 |
| isfinite(const Float6E3M2 &a) | 判断给定的Float6E3M2数值a是否为有限值。 |
| min() | 返回Float6E3M2数据类型的最小正规格化数。 |
| lowest() | 返回Float6E3M2数据类型的最小有限值。 |
| max() | 返回Float6E3M2数据类型的最大有限值。 |
| epsilon() | 返回Float6E3M2数据类型的机器epsilon值。 |
| round_error() | 返回Float6E3M2数据类型的最大舍入误差。 |
| infinity() | 返回Float6E3M2数据类型的最大有限值（该数据类型无无穷大值）。 |
| quiet_NaN() | 返回Float6E3M2数据类型的零值（该数据类型无quiet NaN值）。 |
| signaling_NaN() | 返回Float6E3M2数据类型的零值（该数据类型无signaling NaN值）。 |
| denorm_min() | 返回Float6E3M2数据类型的最小正非规格化值。 |
