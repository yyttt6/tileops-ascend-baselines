# bfloat16

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| from_bits() | Empty structure, which is not used currently.|
| from_bits_t() | Empty structure, which is not used currently.|
| bfloat16(uint16_t bits, [[maybe_unused]] from_bits_t fromBits) | Constructs an instance of `struct bfloat16`.|
| bfloat16() | Constructs an instance of `struct bfloat16`.|
| bfloat16(float v) | Constructs an instance of `struct bfloat16`.|
| bfloat16(const complex64 &val) | Constructs an instance of `struct bfloat16`.|
| bfloat16(const complex128 &val) | Constructs an instance of `struct bfloat16`.|
| bfloat16(const T &val) | Constructs an instance of `struct bfloat16`.|
| float() | Converts the `value` member of `struct bfloat16` from `uint16_t` to `float`.|
| bool() | Checks if the absolute value of a `bfloat16` exceeds the `float` epsilon.|
| epsilon() | Returns the `bfloat16` epsilon.|
| short() | Converts a `bfloat16` value to `short`.|
| int() | Converts a `bfloat16` value to `int`.|
| long() | Converts a `bfloat16` value to `long`.|
| char() | Converts a `bfloat16` value to `char`.|
| double() | Converts a `bfloat16` value to `double`.|
| complex64() | Converts a `bfloat16` value to `complex64`.|
| complex128() | Converts a `bfloat16` value to `complex128`.|
| round_to_bfloat16(float v) | Converts a floating-point number to `bfloat16` using round-nearest-to-even.|
| highest() | Returns the largest representable `bfloat16` value.|
| lowest() | Returns the smallest representable `bfloat16` value.|
| min_positive_normal() | Returns the smallest positive normal value of the `bfloat16` type.|
| IsZero() | Checks if a `bfloat16` value is **0**.|
| float_isnan(const float &x) | Checks if a `float` value *x* is subnormal.|
| isinf(const bfloat16 &a) | Checks if a given `bfloat16` value *a* is infinity.|
| isnan(const bfloat16 &a) | Checks if a given `bfloat16` value *a* is NaN.|
| isfinite(const bfloat16 &a) | Checks if a given `bfloat16` value *a* is finite.|
| exp(const bfloat16 &a) | Returns **e** raised to the power of *a*.|
| log(const bfloat16 &a) | Returns the natural logarithm of *a*.|
| log10(const bfloat16 &a) | Returns the base-10 logarithm of *a*.|
| sqrt(const bfloat16 &a) | Returns the square root of *a*.|
| pow(const bfloat16 &a, const bfloat16 &b) | Returns *a* raised to the power of *b*.|
| sin(const bfloat16 &a) | Returns the sine of angle *a*.|
| cos(const bfloat16 &a) | Returns the cosine of angle *a*.|
| tan(const bfloat16 &a) | Returns the tangent of angle *a*.|
| tanh(const bfloat16 &a) | Returns the hyperbolic tangent of angle *a*.|
| floor(const bfloat16 &a) | Returns the largest integer not greater than `bfloat16` value *a*.|
| ceil(const bfloat16 &a) | Returns the smallest integer not less than `bfloat16` value *a*.|
| min() | Returns the smallest positive normal value of the `bfloat16` type.|
| max() | Returns the largest representable `bfloat16` value.|
| round_error() | Returns the maximum rounding error.|
| infinity() | Returns the infinity value of the `bfloat16` type.|
| quiet_NaN() | Returns the quiet (non-signaling) NaN value of the `bfloat16` type.|
| signaling_NaN() | Returns the signaling NaN value of the `bfloat16` type.|
| denorm_min() | Returns the smallest positive subnormal number of the `bfloat16` type.|
