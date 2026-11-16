# HiFloat4

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| FromBits() | 返回用于从位模式构造HiFloat4对象的标签。 |
| HiFloat4() | struct HiFloat4的默认构造函数，初始化为零值。 |
| HiFloat4(uint8_t bits, FromBitsTag) | struct HiFloat4的构造函数，从原始位模式构造对象。 |
| HiFloat4(float v) | struct HiFloat4的构造函数，从float值构造对象。 |
| operator float() | 将HiFloat4转换为float。 |
| operator double() | 将HiFloat4转换为double。 |
| Epsilon() | 返回HiFloat4数据类型的机器epsilon值。 |
| Highest() | 返回HiFloat4数据类型的最大有限值。 |
| Lowest() | 返回HiFloat4数据类型的最小有限值。 |
| MinPositiveNormal() | 返回HiFloat4数据类型的最小正规格化数。 |
| IsZero() | 判断当前HiFloat4数值是否为零。 |
| IsNaN() | 判断当前HiFloat4数值是否为NaN。 |
| IsInf() | 判断当前HiFloat4数值是否为无穷大。 |
| isinf(const HiFloat4 &a) | 判断给定的HiFloat4数值a是否为无穷大。 |
| isnan(const HiFloat4 &a) | 判断给定的HiFloat4数值a是否为NaN。 |
| isfinite(const HiFloat4 &a) | 判断给定的HiFloat4数值a是否为有限值。 |
| min() | 返回HiFloat4数据类型的最小正规格化数。 |
| lowest() | 返回HiFloat4数据类型的最小有限值。 |
| max() | 返回HiFloat4数据类型的最大有限值。 |
| epsilon() | 返回HiFloat4数据类型的机器epsilon值。 |
| round_error() | 返回HiFloat4数据类型的最大舍入误差。 |
| infinity() | 返回HiFloat4数据类型的最大有限值（该数据类型无无穷大值）。 |
| quiet_NaN() | 返回HiFloat4数据类型的零值（该数据类型无quiet NaN值）。 |
| signaling_NaN() | 返回HiFloat4数据类型的零值（该数据类型无signaling NaN值）。 |
| denorm_min() | 返回HiFloat4数据类型的最小正非规格化值。 |
