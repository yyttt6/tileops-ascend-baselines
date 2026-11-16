# Float8E8M0

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| FromBits() | 返回用于从位模式构造Float8E8M0对象的标签。 |
| Float8E8M0() | struct Float8E8M0的默认构造函数，初始化为零值。 |
| Float8E8M0(uint8_t bits, FromBitsTag) | struct Float8E8M0的构造函数，从原始位模式构造对象。 |
| Float8E8M0(float v) | struct Float8E8M0的构造函数，从float值构造对象。 |
| operator float() | 将Float8E8M0转换为float。 |
| operator double() | 将Float8E8M0转换为double。 |
| Epsilon() | 返回Float8E8M0数据类型的机器epsilon值。 |
| Highest() | 返回Float8E8M0数据类型的最大值。 |
| Lowest() | 返回Float8E8M0数据类型的最小正值（无符号格式，返回最小正值）。 |
| MinPositive() | 返回Float8E8M0数据类型的最小正值。 |
| IsZero() | 判断当前Float8E8M0数值是否为零。 |
| IsNaN() | 判断当前Float8E8M0数值是否为NaN。 |
| IsInf() | 判断当前Float8E8M0数值是否为无穷大。 |
| GetExponent() | 获取当前Float8E8M0数值的无偏指数值。 |
| isinf(const Float8E8M0 &a) | 判断给定的Float8E8M0数值a是否为无穷大。 |
| isnan(const Float8E8M0 &a) | 判断给定的Float8E8M0数值a是否为NaN。 |
| isfinite(const Float8E8M0 &a) | 判断给定的Float8E8M0数值a是否为有限值。 |
| min() | 返回Float8E8M0数据类型的最小正值。 |
| lowest() | 返回Float8E8M0数据类型的最小正值（无符号格式，返回最小正值）。 |
| max() | 返回Float8E8M0数据类型的最大值。 |
| epsilon() | 返回Float8E8M0数据类型的机器epsilon值。 |
| round_error() | 返回Float8E8M0数据类型的最大舍入误差。 |
| infinity() | 返回Float8E8M0数据类型的最大有限值（该数据类型无无穷大值）。 |
| quiet_NaN() | 返回Float8E8M0数据类型的quiet NaN值。 |
| signaling_NaN() | 返回Float8E8M0数据类型的quiet NaN值（该数据类型无signaling NaN值）。 |
| denorm_min() | 返回Float8E8M0数据类型的最小正非规格化值。 |
