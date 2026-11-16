/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "kernel_operator.h"

class IntegralConstantTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IntegralConstantTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IntegralConstantTest TearDownTestCase" << std::endl;
    }
};

// test the fundamental properties of integral_constant
TEST(IntegralConstantTest, BasicProperties) {
    using IntTrue = AscendC::Std::integral_constant<int, 1>;
    using IntFalse = AscendC::Std::integral_constant<int, 0>;
    using CharA = AscendC::Std::integral_constant<char, 'A'>;

    // test the value of a static const
    EXPECT_EQ(IntTrue::value, 1);
    EXPECT_EQ(IntFalse::value, 0);
    EXPECT_EQ(CharA::value, 'A');

    // test () operator overloading
    EXPECT_EQ(IntTrue()(), 1);
    EXPECT_EQ(IntFalse()(), 0);
    EXPECT_EQ(CharA()(), 'A');

    // test type definition
    EXPECT_TRUE((AscendC::Std::is_same_v<typename IntTrue::value_type, int>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename IntTrue::type, IntTrue>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename CharA::value_type, char>));
}

// testing the boolean specialization of integral_constant
TEST(IntegralConstantTest, BooleanSpecialization) {
    using TrueType = AscendC::Std::true_type;
    using FalseType = AscendC::Std::false_type;

    // test the value of a static const
    EXPECT_TRUE(TrueType::value);
    EXPECT_FALSE(FalseType::value);

    // test () operator overloading
    EXPECT_TRUE(TrueType()());
    EXPECT_FALSE(FalseType()());

    // test type definition
    EXPECT_TRUE((AscendC::Std::is_same_v<typename TrueType::value_type, bool>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename TrueType::type, TrueType>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename FalseType::type, FalseType>));
}

// implementing simple template conditions using integral_constant
template <typename T>
struct IsIntegralConstant : AscendC::Std::false_type {};

template <typename T, T v>
struct IsIntegralConstant<AscendC::Std::integral_constant<T, v>> : AscendC::Std::true_type {};

// testing the use of integral_constant in template metaprogramming
TEST(IntegralConstantTest, TemplateMetaprogramming) {
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::integral_constant<int, 42>>::value));
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::true_type>::value));
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::false_type>::value));
    EXPECT_FALSE((IsIntegralConstant<int>::value));
    EXPECT_FALSE((IsIntegralConstant<std::string>::value));

    // tesing uses integral_constant for static assertions
    static_assert(AscendC::Std::integral_constant<int, 10>::value == 10, "Should be 10");
    static_assert(AscendC::Std::integral_constant<bool, true>::value, "Should be true");
}

// test integral_constant as a function parameter
TEST(IntegralConstantTest, AsFunctionParameter) {
    // a function that accepts integral_constant as a parameter
    auto checkValue = [](auto constant) {
        return constant.value;
    };

    EXPECT_EQ(checkValue(AscendC::Std::integral_constant<int, 5>{}), 5);
    EXPECT_EQ(checkValue(AscendC::Std::integral_constant<char, 'Z'>{}), 'Z');
    EXPECT_EQ(checkValue(AscendC::Std::true_type{}), true);
    EXPECT_EQ(checkValue(AscendC::Std::false_type{}), false);

    // function with conditional branching
    auto conditionalFunction = [](auto cond) {
        if constexpr (cond.value) {
            return 100;
        } else {
            return 200;
        }
    };

    EXPECT_EQ(conditionalFunction(AscendC::Std::true_type{}), 100);
    EXPECT_EQ(conditionalFunction(AscendC::Std::false_type{}), 200);
}

// test basic attributes
TEST(IntegralConstantTest, IntBasicProperties) {
    using Zero = AscendC::Std::Int<0>;
    using One = AscendC::Std::Int<1>;
    using Large = AscendC::Std::Int<0xFFFFFFFF>;

    // verify the value of a static const
    EXPECT_EQ(Zero::value, 0);
    EXPECT_EQ(One::value, 1);
    EXPECT_EQ(Large::value, 0xFFFFFFFF);

    // validation type definition
    EXPECT_TRUE((AscendC::Std::is_same_v<typename Zero::value_type, size_t>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename Zero::type, Zero>));
    EXPECT_TRUE((AscendC::Std::is_same_v<Zero, AscendC::Std::integral_constant<size_t, 0>>));

    // verify () operator overloading
    EXPECT_EQ(Zero()(), 0);
    EXPECT_EQ(One()(), 1);
    EXPECT_EQ(Large()(), 0xFFFFFFFF);
}

// test compile-time calculations
TEST(IntegralConstantTest, CompileTimeOperations) {
    // add
    static_assert((AscendC::Std::Int<5>::value + AscendC::Std::Int<3>::value) == 8, "Addition failed");
    
    // muls
    static_assert((AscendC::Std::Int<4>::value * AscendC::Std::Int<6>::value) == 24, "Multiplication failed");
    
    // compare
    static_assert(AscendC::Std::Int<10>::value > AscendC::Std::Int<5>::value, "Comparison failed");
    static_assert(AscendC::Std::Int<7>::value != AscendC::Std::Int<77>::value, "Equality check failed");
}

// template metaprogramming: calculting factorials
template <typename N>
struct Factorial : AscendC::Std::Int<N::value * Factorial<AscendC::Std::Int<N::value - 1>>::value> {};

template <>
struct Factorial<AscendC::Std::Int<0>> : AscendC::Std::Int<1> {};

// template metaprogramming: condition selection
template <typename Condition, typename Then, typename Else>
struct IfThenElse : Then {};

template <typename Then, typename Else>
struct IfThenElse<AscendC::Std::Int<0>, Then, Else> : Else {};

// testing as a template parameter
TEST(IntegralConstantTest, TemplateParameter) {
    // verifying compile-time calcultion result
    static_assert(Factorial<AscendC::Std::Int<0>>::value == 1, "Factorial(0) failed");
    static_assert(Factorial<AscendC::Std::Int<1>>::value == 1, "Factorial(1) failed");
    static_assert(Factorial<AscendC::Std::Int<5>>::value == 120, "Factorial(5) failed");

    // verifying condition selection
    using Result1 = IfThenElse<AscendC::Std::Int<1>, AscendC::Std::Int<100>, AscendC::Std::Int<200>>;
    using Result2 = IfThenElse<AscendC::Std::Int<0>, AscendC::Std::Int<100>, AscendC::Std::Int<200>>;
    EXPECT_EQ(Result1::value, 100);
    EXPECT_EQ(Result2::value, 200);
}

// verifying of Compatibility with AscendC::Std::enable_if
template <typename T>
auto getValue(T t, AscendC::Std::enable_if_t<T::value <= 10, int>* = nullptr) {
    return t.value * 2;
}

template <typename T>
auto getValue(T t, AscendC::Std::enable_if_t<T::value >= 10, int>* = nullptr) {
    return t.value / 2;
}

// testing combined with standard library features
TEST(IntegralConstantTest, StdTraitsIntegration) {
    // verifying of Compatibility with AscendC::Std::is_same
    EXPECT_TRUE((AscendC::Std::is_same_v<AscendC::Std::Int<42>, AscendC::Std::integral_constant<size_t, 42>>));
    EXPECT_FALSE((AscendC::Std::is_same_v<AscendC::Std::Int<1>, AscendC::Std::Int<2>>));

    // verifying of Compatibility with AscendC::Std::conditional
    using Result = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::Std::Int<1>, AscendC::Std::Int<1>>,
        AscendC::Std::Int<100>,
        AscendC::Std::Int<200>
    >;
    EXPECT_EQ(Result::value, 100);

    EXPECT_EQ(getValue(AscendC::Std::Int<5>{}), 10);
    EXPECT_EQ(getValue(AscendC::Std::Int<20>{}), 10);
}

// application scenarios during test run
TEST(IntegralConstantTest, RuntimeUsage) {
    // function parameter
    auto add = [](auto a, auto b) {
        return a.value + b.value;
    };
    
    EXPECT_EQ(add(AscendC::Std::Int<3>{}, AscendC::Std::Int<7>{}), 10);

    // array size
    std::array<int, AscendC::Std::Int<5>::value> arr;
    EXPECT_EQ(arr.size(), 5);

    // loop times
    int sum = 0;
    for (size_t i = 0; i < AscendC::Std::Int<4>::value; ++i) {
        sum += i;
    }
    EXPECT_EQ(sum, 6); // 0+1+2+3=6
}