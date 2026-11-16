# HiFloat8

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| FromBits() | 返回用于从位模式构造HiFloat8对象的标签。 |
| HiFloat8() | struct HiFloat8的默认构造函数，初始化为零值。 |
| HiFloat8(uint8_t bits, FromBitsTag) | struct HiFloat8的构造函数，从原始位模式构造对象。 |
| HiFloat8(float f32) | struct HiFloat8的构造函数，从float值构造对象。 |
| operator float() | 将HiFloat8转换为float。 |
| IsNaN() | 判断当前HiFloat8数值是否为NaN。 |
| IsInf() | 判断当前HiFloat8数值是否为无穷大。 |
| IsZero() | 判断当前HiFloat8数值是否为零。 |
| BitsFromFp32(uint32_t f32) | 将FP32位模式转换为HiFloat8位模式。 |
| isinf(const HiFloat8 &a) | 判断给定的HiFloat8数值a是否为无穷大。 |
| isnan(const HiFloat8 &a) | 判断给定的HiFloat8数值a是否为NaN。 |
| isfinite(const HiFloat8 &a) | 判断给定的HiFloat8数值a是否为有限值。 |
| min() | 返回HiFloat8数据类型的最小正规格化数。 |
| lowest() | 返回HiFloat8数据类型的最小有限值。 |
| max() | 返回HiFloat8数据类型的最大值。 |
| epsilon() | 返回HiFloat8数据类型的机器epsilon值。 |
| round_error() | 返回HiFloat8数据类型的最大舍入误差。 |
| infinity() | 返回HiFloat8数据类型的无穷大值。 |
| quiet_NaN() | 返回HiFloat8数据类型的quiet NaN值。 |
| signaling_NaN() | 返回HiFloat8数据类型的signaling NaN值。 |
| denorm_min() | 返回HiFloat8数据类型的最小正非规格化值。 |
