# fp16\_t

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| tagFp16(void) | Constructs a default instance of `fp16_t` (without any parameters).|
| tagFp16(const T &value) | Constructs an instance of `fp16_t` (with a parameter of any data type).|
| tagFp16(const bfloat16& value) | Constructs an instance of `fp16_t` (with a parameter of the `bfloat16` type).|
| tagFp16(const uint16_t &uiVal) | Constructs an instance of `fp16_t` (with a parameter of the `uint16_t` type).|
| tagFp16(const tagFp16 &fp) | Constructs an instance of `fp16_t` (with a parameter of the `fp16_t` type). It is a copy constructor.|
| float() | Converts `fp16_t` to `float(fp32)`.|
| bfloat16() | Converts `fp16_t` to `bfloat16`.|
| double() | Converts `fp16_t` to `double (fp64)`.|
| int8_t() | Converts `fp16_t` to `int8_t`.|
| uint8_t() | Converts `fp16_t` to `uint8_t`.|
| int16_t() | Converts `fp16_t` to `int16_t`.|
| uint16_t() | Converts `fp16_t` to `uint16_t`.|
| int32_t() | Converts `fp16_t` to `int32_t`.|
| uint32_t() | Converts `fp16_t` to `uint32_t`.|
| int64_t() | Converts `fp16_t` to `int64_t`.|
| uint64_t() | Converts `fp16_t` to `uint64_t`.|
| bool() | Converts `fp16_t` to `bool`.|
| IsInf() | Checks if the value of `fp16_t` is infinite. **1** is returned for positive infinity, **-1** for negative infinity, and **0** otherwise.|
| toFloat() | Converts `fp16_t` to `float(fp32)`.|
| toDouble() | Converts `fp16_t` to `double(fp64)`.|
| toInt8() | Converts `fp16_t` to `int8_t`.|
| toUInt8() | Converts `fp16_t` to `uint8_t`.|
| toInt16() | Converts `fp16_t` to `int16_t`.|
| toUInt16() | Converts `fp16_t` to `uint16_t`.|
| toInt32() | Converts `fp16_t` to `int32_t`.|
| toUInt32() | Converts `fp16_t` to `uint32_t`.|
| ExtractFP16(const uint16_t &val, uint16_t *s, int16_t*e, uint16_t *m) | Extracts the sign, exponent, and mantissa of an `fp16_t` value.|
| ReverseMan(bool negative, T *man) | Computes the two's complement of the mantissa when the sign bit is negative.|
| MinMan(const int16_t &ea, T *ma, const int16_t &eb, T*mb) | Selects the mantissa of the smaller exponent and right-shifts it for alignment.|
| RightShift(T man, int16_t shift) | Right-shifts the mantissa by *shift* bits.|
| GetManSum(int16_t ea, const T &ma, int16_t eb, const T &mb) | Computes the sum of the mantissas of two `fp16_t` values. The template parameter T can be uint16_t, uint32_t, or uint64_t.|
| ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0) | Rounds the mantissa of an `fp16_t` or `float` value to the nearest value.|
| GetManBitLength(T man) | Obtains the bit length of the mantissa of a floating-point value.|
| isnan(op::fp16_t value) | Checks if a value is NaN.|
