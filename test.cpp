#include "bit_permutations.hpp"
#include "bit_permutations_naive.hpp"

#define CXX26_BIT_PERMUTATIONS_ENABLE_STATIC_TESTS 1

#include <iostream>
#include <random>
#include <source_location>

using namespace cxx26bp;
using namespace cxx26bp::detail;

namespace {

// x: 0001 1001
// m: 0111 0111

// a: 0001 0001
// n: 0011 0001
// e: 0011 0001

[[maybe_unused]] void assert_fail(const char* expr,
                                  std::source_location loc = std::source_location::current())
{
    std::cerr << loc.function_name() << "\n\n    " << expr << '\n';
    std::exit(1);
}

#define ASSERT(...) ((__VA_ARGS__) ? void() : assert_fail(#__VA_ARGS__))

#if CXX26_BIT_PERMUTATIONS_ENABLE_STATIC_TESTS && !defined(__INTELLISENSE__)
#define ASSERT_S(...)                                                                              \
    static_assert(__VA_ARGS__);                                                                    \
    ASSERT(__VA_ARGS__)
#else
#define ASSERT_S(...) ASSERT(__VA_ARGS__)
#endif

void test_is_pow2_or_zero()
{
    ASSERT_S(is_pow2_or_zero(0));
    ASSERT_S(is_pow2_or_zero(1));
    ASSERT_S(is_pow2_or_zero(2));
    ASSERT_S(!is_pow2_or_zero(3));
    ASSERT_S(is_pow2_or_zero(4));
}

void test_log2_floor()
{
    ASSERT_S(log2_floor(0) == 0);
    ASSERT_S(log2_floor(1) == 0);
    ASSERT_S(log2_floor(2) == 1);
    ASSERT_S(log2_floor(3) == 1);
    ASSERT_S(log2_floor(4) == 2);
    ASSERT_S(log2_floor(5) == 2);
    ASSERT_S(log2_floor(6) == 2);
    ASSERT_S(log2_floor(7) == 2);
    ASSERT_S(log2_floor(8) == 3);
}

void test_log2_ceil()
{
    ASSERT_S(log2_ceil(0) == 0);
    ASSERT_S(log2_ceil(1) == 0);
    ASSERT_S(log2_ceil(2) == 1);
    ASSERT_S(log2_ceil(3) == 2);
    ASSERT_S(log2_ceil(4) == 2);
    ASSERT_S(log2_ceil(5) == 3);
    ASSERT_S(log2_ceil(6) == 3);
    ASSERT_S(log2_ceil(7) == 3);
    ASSERT_S(log2_ceil(8) == 3);
}

void test_alternate01()
{
    ASSERT_S(alternate01<std::uint8_t>(1, 1) == 0b0101'0101);
    ASSERT_S(alternate01<std::uint8_t>(2, 2) == 0b0011'0011);
    ASSERT_S(alternate01<std::uint8_t>(3, 3) == 0b1100'0111);
    ASSERT_S(alternate01<std::uint8_t>(4, 4) == 0b0000'1111);
    ASSERT_S(alternate01<std::uint8_t>(0, 8) == 0b1111'1111);

    ASSERT_S(alternate01<std::uint32_t>(1, 1) == 0x5555'5555);
    ASSERT_S(alternate01<std::uint32_t>(2, 2) == 0x3333'3333);
    ASSERT_S(alternate01<std::uint32_t>(3, 3) == 0xc71c'71c7);
    ASSERT_S(alternate01<std::uint32_t>(4, 4) == 0x0f0f'0f0f);
    ASSERT_S(alternate01<std::uint32_t>(8, 8) == 0x00ff'00ff);

    ASSERT_S(alternate01<std::uint8_t>(2, 1) == 0b0100'1001);
    ASSERT_S(alternate01<std::uint8_t>(3, 1) == 0b0001'0001);
    ASSERT_S(alternate01<std::uint8_t>(1, 2) == 0b1101'1011);
    ASSERT_S(alternate01<std::uint8_t>(3, 2) == 0b0110'0011);
    ASSERT_S(alternate01<std::uint8_t>(1, 3) == 0b0111'0111);
    ASSERT_S(alternate01<std::uint8_t>(2, 3) == 0b1110'0111);

    ASSERT_S(alternate01<std::uint32_t>(3, 1) == 0x1111'1111);
    ASSERT_S(alternate01<std::uint32_t>(1, 3) == 0x7777'7777);
    ASSERT_S(alternate01<std::uint32_t>(8, 4) == 0x0f00'f00f);
    ASSERT_S(alternate01<std::uint32_t>(12, 4) == 0x000f'000f);
}

template <std::uint32_t (&F)(std::uint32_t, int)>
void test_repeat_bits()
{
    ASSERT_S(F(0b0, 1) == 0x0);
    ASSERT_S(F(0b0, 3) == 0x0);
    ASSERT_S(F(0b0, 32) == 0x0);

    ASSERT_S(F(0b01, 2) == 0x5555'5555);
    ASSERT_S(F(0x3, 4) == 0x3333'3333);
    ASSERT_S(F(0b000111, 6) == 0xc71c'71c7);
    ASSERT_S(F(0x0f, 8) == 0x0f0f'0f0f);
    ASSERT_S(F(0x00ff, 16) == 0x00ff'00ff);

    ASSERT_S(F(0b0001, 4) == 0x1111'1111);
    ASSERT_S(F(0b0111, 4) == 0x7777'7777);
    ASSERT_S(F(0x00f, 12) == 0x0f00'f00f);
    ASSERT_S(F(0x000f, 16) == 0x000f'000f);
}

template <int (&F)(unsigned)>
void test_popcount()
{
    ASSERT_S(F(0) == 0);
    ASSERT_S(F(-1u) == detail::digits_v<unsigned>);
    ASSERT_S(F(0b10010100111u) == 6);
}

template <int (&F)(std::uint8_t)>
void test_countr_zero()
{
    ASSERT_S(F(0) == 8);
    ASSERT_S(F(1) == 0);
    ASSERT_S(F(2) == 1);
    ASSERT_S(F(3) == 0);
    ASSERT_S(F(4) == 2);
    ASSERT_S(F(5) == 0);
    ASSERT_S(F(6) == 1);
    ASSERT_S(F(7) == 0);
    ASSERT_S(F(8) == 3);

    ASSERT_S(F(128) == 7);
    ASSERT_S(F(129) == 0);
    ASSERT_S(F(std::uint8_t(-1)) == 0);
}

template <int (&F)(std::uint8_t)>
void test_countl_zero()
{
    ASSERT_S(F(0) == 8);
    ASSERT_S(F(1) == 7);
    ASSERT_S(F(2) == 6);
    ASSERT_S(F(3) == 6);
    ASSERT_S(F(4) == 5);
    ASSERT_S(F(5) == 5);
    ASSERT_S(F(6) == 5);
    ASSERT_S(F(7) == 5);
    ASSERT_S(F(8) == 4);

    ASSERT_S(F(128) == 0);
    ASSERT_S(F(129) == 0);
    ASSERT_S(F(std::uint8_t(-1)) == 0);
}

template <std::uint8_t (&F)(std::uint8_t)>
void test_bipp()
{
    ASSERT_S(F(0b0000'0000u) == 0b0000'0000);
    ASSERT_S(F(0b1111'1111u) == 0b0101'0101);
    ASSERT_S(F(0b1001'0000u) == 0b0111'0000);
    ASSERT_S(F(0b0100'1000u) == 0b0011'1000);
}

template <std::uint8_t (&F)(std::uint8_t)>
void test_reverse_bits()
{
    ASSERT_S(F(std::uint8_t { 0b1101'1001u }) == 0b1001'1011);
    ASSERT_S(F(std::uint8_t { 0b1101'1001u }) == 0b1001'1011);
}

template <std::uint8_t (&F)(std::uint8_t, std::uint8_t)>
void test_compress_bitsr()
{
    ASSERT_S(F(0b000000u, 0b101010u) == 0b000);
    ASSERT_S(F(0b010101u, 0b101010u) == 0b000);
    ASSERT_S(F(0b101010u, 0b101010u) == 0b111);
    ASSERT_S(F(0b111111u, 0b101010u) == 0b111);
}

template <std::uint8_t (&F)(std::uint8_t, std::uint8_t)>
void test_compress_bitsl()
{
    ASSERT_S(F(0b000000u, 0b101010u) == 0b000'00000);
    ASSERT_S(F(0b010101u, 0b101010u) == 0b000'00000);
    ASSERT_S(F(0b101010u, 0b101010u) == 0b111'00000);
    ASSERT_S(F(0b111111u, 0b101010u) == 0b111'00000);
}

template <std::uint8_t (&F)(std::uint8_t, std::uint8_t)>
void test_expand_bitsr()
{
    ASSERT_S(F(0b000u, 0b101010u) == 0b000000);
    ASSERT_S(F(0b101u, 0b101011u) == 0b001001);
    ASSERT_S(F(0b101u, 0b101010u) == 0b100010);
    ASSERT_S(F(0b111u, 0b101010u) == 0b101010);
}

template <std::uint8_t (&F)(std::uint8_t)>
void text_next_bit_permutation()
{
    ASSERT_S(F(0) == 0);
    ASSERT_S(F(1) == 2);
    ASSERT_S(F(2) == 4);
    ASSERT_S(F(3) == 5);
    ASSERT_S(F(4) == 8);
    ASSERT_S(F(5) == 6);
    ASSERT_S(F(6) == 9);
    ASSERT_S(F(7) == 11);

    ASSERT_S(F(8) == 16);
    ASSERT_S(F(9) == 10);
    ASSERT_S(F(10) == 12);
    ASSERT_S(F(11) == 13);
    ASSERT_S(F(12) == 17);
    ASSERT_S(F(13) == 14);
    ASSERT_S(F(14) == 19);
    ASSERT_S(F(15) == 23);

    ASSERT_S(F(128) == 0);
    ASSERT_S(F(224) == 0);
}

template <std::uint8_t (&F)(std::uint8_t)>
void text_prev_bit_permutation()
{
    ASSERT_S(F(0) == 0);
    ASSERT_S(F(2) == 1);
    ASSERT_S(F(4) == 2);
    ASSERT_S(F(5) == 3);
    ASSERT_S(F(8) == 4);
    ASSERT_S(F(6) == 5);
    ASSERT_S(F(9) == 6);
    ASSERT_S(F(11) == 7);

    ASSERT_S(F(16) == 8);
    ASSERT_S(F(10) == 9);
    ASSERT_S(F(12) == 10);
    ASSERT_S(F(13) == 11);
    ASSERT_S(F(17) == 12);
    ASSERT_S(F(14) == 13);
    ASSERT_S(F(19) == 14);
    ASSERT_S(F(23) == 15);

    ASSERT_S(F(1) == 0);
    ASSERT_S(F(7) == 0);
}

constexpr int seed = 0x12345;
#ifndef FUZZ_COUNT
#ifdef NDEBUG
constexpr int default_fuzz_count = 1024 * 1024 * 16;
#else
constexpr int default_fuzz_count = 1024 * 1024;
#endif
#else
constexpr int default_fuzz_count = FUZZ_COUNT;
#endif
using rng_type = std::mt19937_64;
using distr_type = std::uniform_int_distribution<uint64_t>;

template <typename T>
constexpr T rand_int(rng_type& rng, distr_type& d)
{
    constexpr int N = detail::digits_v<T>;

    // Instead of generating completely random integers, we randomly pick special cases in 6/8
    // instances.
    switch (d(rng) & 0b111) {
    case 0:
        return 0;
    case 1:
        return 1;
    case 2:
        return static_cast<T>(-1);
    case 3:
        return alternate01<T>(1, 1);
    case 4:
        return alternate01<T>(N / 2, N / 2);
    case 5:
        return static_cast<T>(~alternate01<T>(1, 1));
    case 6:
        return static_cast<T>(~alternate01<T>(N / 2, N / 2));
    default:; // not exhaustive
    }

    if constexpr (N <= 64) {
        return static_cast<uint64_t>(d(rng));
    }
    else {
        T result = 0;
        for (int i = 0; i < N; i += 64) {
            result <<= 64;
            result |= d(rng);
        }
        return result;
    }
}

template <typename T, T (&Fun)(T), T (&Naive)(T), int FuzzCount = default_fuzz_count>
void naive_fuzz_1()
{
    rng_type rng { seed };
    std::uniform_int_distribution<uint64_t> d;

    for (int i = 0; i < FuzzCount; ++i) {
        const T x = rand_int<T>(rng, d);
        const T actual = Fun(x);
        const T naive = Naive(x);
        ASSERT(actual == naive);
    }
}

template <typename T, int (&Fun)(T), int (&Naive)(T), int FuzzCount = default_fuzz_count>
void naive_fuzz_int()
{
    rng_type rng { seed };
    std::uniform_int_distribution<uint64_t> d;

    for (int i = 0; i < FuzzCount; ++i) {
        const int x = rand_int<T>(rng, d);
        const int actual = Fun(x);
        const int naive = Naive(x);
        ASSERT(actual == naive);
    }
}

template <typename T, T (&Fun)(T, T), T (&Naive)(T, T), int FuzzCount = default_fuzz_count>
void naive_fuzz_2()
{
    rng_type rng { seed };
    std::uniform_int_distribution<uint64_t> d;

    for (int i = 0; i < FuzzCount; ++i) {
        const T x = rand_int<T>(rng, d);
        const T m = rand_int<T>(rng, d);
        const T actual = Fun(x, m);
        const T naive = Naive(x, m);
        ASSERT(actual == naive);
    }
}

template <typename T, T (&Fun)(T, int), T (&Naive)(T, int), int FuzzCount = default_fuzz_count>
void naive_fuzz_repeat_bits()
{
    rng_type rng { seed };
    std::uniform_int_distribution<uint64_t> d;
    std::uniform_int_distribution<int> d_length { 1, detail::digits_v<T> };

    for (int i = 0; i < FuzzCount; ++i) {
        const T x = rand_int<T>(rng, d);
        const int length = d_length(rng);
        const T actual = Fun(x, length);
        const T naive = Naive(x, length);
        ASSERT(actual == naive);
    }
}

#ifdef CXX26_BIT_PERMUTATIONS_U128
#define IF_U128(...) __VA_ARGS__
#else
#define IF_U128(...)
#endif

#ifdef CXX26_BIT_PERMUTATIONS_BITINT
#define IF_BITINT(...) __VA_ARGS__
#else
#define IF_BITINT(...)
#endif

#define FUZZ_1(T, f) naive_fuzz_1<T, f<T>, f##_naive<T>>
#define FUZZ_2(T, f) naive_fuzz_2<T, f<T>, f##_naive<T>>
#define FUZZ_INT(T, f) naive_fuzz_int<T, f<T>, f##_naive<T>>
#define FUZZ_REPEAT(T, f) naive_fuzz_repeat_bits<T, f<T>, f##_naive<T>>

// clang-format off
constexpr void (*tests[])() = {
    test_is_pow2_or_zero,
    test_log2_floor,
    test_log2_ceil,
    test_alternate01,

#ifndef __INTELLISENSE__
    test_repeat_bits<repeat_bits>,
    test_repeat_bits<repeat_bits_naive>,

    test_popcount<popcount>,
    test_popcount<popcount_naive>,

    test_countl_zero<countl_zero>,
    test_countl_zero<countl_zero_naive>,

    test_countr_zero<countr_zero>,
    test_countr_zero<countr_zero_naive>,

    test_bipp<bitwise_inclusive_right_parity>,
    test_bipp<bitwise_inclusive_right_parity_naive>,
    
    test_reverse_bits<reverse_bits>,
    test_reverse_bits<reverse_bits_naive>,

    test_compress_bitsr<compress_bitsr>,
    test_compress_bitsr<compress_bitsr_naive>,

    test_compress_bitsl<compress_bitsl>,
    test_compress_bitsl<compress_bitsl_naive>,

    test_expand_bitsr<expand_bitsr>,
    test_expand_bitsr<expand_bitsr_naive>,

    text_next_bit_permutation<next_bit_permutation>,
    text_next_bit_permutation<next_bit_permutation_naive>,

    text_prev_bit_permutation<prev_bit_permutation>,
    text_prev_bit_permutation<prev_bit_permutation_naive>,
    
    FUZZ_REPEAT(std::uint8_t,  repeat_bits),
    FUZZ_REPEAT(std::uint16_t, repeat_bits),
    FUZZ_REPEAT(std::uint32_t, repeat_bits),
    FUZZ_REPEAT(std::uint64_t, repeat_bits),
    IF_U128(FUZZ_REPEAT(detail::uint128_t, repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(2), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(3), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(4), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(5), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(6), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(7), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(8), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(200), repeat_bits),)
    IF_BITINT(FUZZ_REPEAT(unsigned _BitInt(256), repeat_bits),)

    FUZZ_INT(std::uint8_t,  popcount),
    FUZZ_INT(std::uint16_t, popcount),
    FUZZ_INT(std::uint32_t, popcount),
    FUZZ_INT(std::uint64_t, popcount),
    IF_U128(FUZZ_INT(detail::uint128_t, popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(2), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(3), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(4), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(5), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(6), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(7), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(8), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(200), popcount),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(256), popcount),)

    FUZZ_INT(std::uint8_t,  countl_zero),
    FUZZ_INT(std::uint16_t, countl_zero),
    FUZZ_INT(std::uint32_t, countl_zero),
    FUZZ_INT(std::uint64_t, countl_zero),
    IF_U128(FUZZ_INT(detail::uint128_t, countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(2), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(3), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(4), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(5), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(6), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(7), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(8), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(200), countl_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(256), countl_zero),)

    FUZZ_INT(std::uint8_t,  countr_zero),
    FUZZ_INT(std::uint16_t, countr_zero),
    FUZZ_INT(std::uint32_t, countr_zero),
    FUZZ_INT(std::uint64_t, countr_zero),
    IF_U128(FUZZ_INT(detail::uint128_t, countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(2), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(3), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(4), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(5), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(6), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(7), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(8), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(200), countr_zero),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(256), countr_zero),)

    FUZZ_INT(std::uint8_t,  countl_one),
    FUZZ_INT(std::uint16_t, countl_one),
    FUZZ_INT(std::uint32_t, countl_one),
    FUZZ_INT(std::uint64_t, countl_one),
    IF_U128(FUZZ_INT(detail::uint128_t, countl_one),)
    IF_U128(FUZZ_INT(detail::uint128_t, countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(2), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(3), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(4), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(5), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(6), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(7), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(8), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(200), countl_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(256), countl_one),)

    FUZZ_INT(std::uint8_t,  countr_one),
    FUZZ_INT(std::uint16_t, countr_one),
    FUZZ_INT(std::uint32_t, countr_one),
    FUZZ_INT(std::uint64_t, countr_one),
    IF_U128(FUZZ_INT(detail::uint128_t, countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(2), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(3), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(4), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(5), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(6), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(7), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(8), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(200), countr_one),)
    IF_BITINT(FUZZ_INT(unsigned _BitInt(256), countr_one),)

    FUZZ_1(std::uint8_t,  reverse_bits),
    FUZZ_1(std::uint16_t, reverse_bits),
    FUZZ_1(std::uint32_t, reverse_bits),
    FUZZ_1(std::uint64_t, reverse_bits),
    IF_U128(FUZZ_1(detail::uint128_t, reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(2), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(3), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(4), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(5), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(6), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(7), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(8), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(200), reverse_bits),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(256), reverse_bits),)

    FUZZ_1(std::uint8_t,  bitwise_inclusive_right_parity),
    FUZZ_1(std::uint16_t, bitwise_inclusive_right_parity),
    FUZZ_1(std::uint32_t, bitwise_inclusive_right_parity),
    FUZZ_1(std::uint64_t, bitwise_inclusive_right_parity),
    IF_U128(FUZZ_1(detail::uint128_t, bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(2), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(3), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(4), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(5), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(6), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(7), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(8), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(200), bitwise_inclusive_right_parity),)
    IF_BITINT(FUZZ_1(unsigned _BitInt(256), bitwise_inclusive_right_parity),)


    // TODO: Large fuzz-testing these against a naive implementation is not viable
    //       because the algorithm has exponential complexity.
    FUZZ_1(std::uint8_t,  next_bit_permutation),
    FUZZ_1(std::uint8_t,  prev_bit_permutation),

    FUZZ_2(std::uint8_t,  compress_bitsr),
    FUZZ_2(std::uint16_t, compress_bitsr),
    FUZZ_2(std::uint32_t, compress_bitsr),
    FUZZ_2(std::uint64_t, compress_bitsr),
    IF_U128(FUZZ_2(detail::uint128_t, compress_bitsr)),
    IF_BITINT(FUZZ_2(unsigned _BitInt(2), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(3), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(4), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(5), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(6), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(7), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(8), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(200), compress_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(256), compress_bitsr),)

    FUZZ_2(std::uint8_t,  compress_bitsl),
    FUZZ_2(std::uint16_t, compress_bitsl),
    FUZZ_2(std::uint32_t, compress_bitsl),
    FUZZ_2(std::uint64_t, compress_bitsl),
    IF_U128(FUZZ_2(detail::uint128_t, compress_bitsl)),
    IF_BITINT(FUZZ_2(unsigned _BitInt(2), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(3), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(4), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(5), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(6), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(7), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(8), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(200), compress_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(256), compress_bitsl),)

    FUZZ_2(std::uint8_t,  expand_bitsr),
    FUZZ_2(std::uint16_t, expand_bitsr),
    FUZZ_2(std::uint32_t, expand_bitsr),
    FUZZ_2(std::uint64_t, expand_bitsr),
    IF_U128(FUZZ_2(detail::uint128_t, expand_bitsr)),
    IF_BITINT(FUZZ_2(unsigned _BitInt(2), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(3), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(4), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(5), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(6), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(7), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(8), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(200), expand_bitsr),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(256), expand_bitsr),)

    FUZZ_2(std::uint8_t,  expand_bitsl),
    FUZZ_2(std::uint16_t, expand_bitsl),
    FUZZ_2(std::uint32_t, expand_bitsl),
    FUZZ_2(std::uint64_t, expand_bitsl),
    IF_U128(FUZZ_2(detail::uint128_t, expand_bitsl)),
    IF_BITINT(FUZZ_2(unsigned _BitInt(2), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(3), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(4), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(5), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(6), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(7), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(8), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(200), expand_bitsl),)
    IF_BITINT(FUZZ_2(unsigned _BitInt(256), expand_bitsl),)
#endif
};
// clang-format on

} // namespace

int main()
{
    std::cout << '[' << std::string(std::size(tests), ' ') << "]\r[" << std::flush;

    for (void (*test)() : tests) {
        test();
        std::cout << '=' << std::flush;
    }
    std::cout << "]\n";
}