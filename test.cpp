#include "bit_permutations.hpp"

#define CXX26_BIT_PERMUTATIONS_ENABLE_STATIC_TESTS 1

#include <iostream>
#include <random>
#include <source_location>

using namespace std::experimental;
using namespace std::experimental::detail;

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
    ASSERT_S(alternate01<std::uint8_t>(1) == 0b1010'1010);
    ASSERT_S(alternate01<std::uint8_t>(2) == 0b1100'1100);
    ASSERT_S(alternate01<std::uint8_t>(3) == 0b0011'1000);
    ASSERT_S(alternate01<std::uint8_t>(4) == 0b1111'0000);
    ASSERT_S(alternate01<std::uint8_t>(8) == 0b0000'0000);

    ASSERT_S(alternate01<std::uint32_t>(1) == 0xaaaa'aaaa);
    ASSERT_S(alternate01<std::uint32_t>(2) == 0xcccc'cccc);
    ASSERT_S(alternate01<std::uint32_t>(3) == 0x38e3'8e38);
    ASSERT_S(alternate01<std::uint32_t>(4) == 0xf0f0'f0f0);
    ASSERT_S(alternate01<std::uint32_t>(8) == 0xff00'ff00);
}

void test_bipp()
{
    ASSERT_S(bitwise_inclusive_prefix_parity(0b0000'0000u) == 0b0000'0000);
    ASSERT_S(bitwise_inclusive_prefix_parity(0b1111'1111u) == 0b0101'0101);
    ASSERT_S(bitwise_inclusive_prefix_parity(0b1001'0000u) == 0b0111'0000);
    ASSERT_S(bitwise_inclusive_prefix_parity(0b0100'1000u) == 0b0011'1000);

    ASSERT_S(bitwise_inclusive_prefix_parity(0x0123'4567'89ab'cdefu)
             == bitwise_inclusive_prefix_parity_naive(0x0123'4567'89ab'cdefu));
}

void test_reverse_bits()
{
    ASSERT_S(reverse_bits(std::uint8_t { 0b1101'1001u }) == 0b1001'1011);
    ASSERT_S(reverse_bits(std::uint8_t { 0b1101'1001u }) == 0b1001'1011);
    ASSERT_S(reverse_bits(0x0123'4567'89ab'cdefu)
             == detail::reverse_bits_naive(0x0123'4567'89ab'cdefu));
}

template <std::uint8_t (&F)(std::uint8_t, std::uint8_t)>
void test_compress_bitsr()
{
    ASSERT_S(F(0b000000u, 0b101010u) == 0b000);
    ASSERT_S(F(0b010101u, 0b101010u) == 0b000);
    ASSERT_S(F(0b101010u, 0b101010u) == 0b111);
    ASSERT_S(F(0b111111u, 0b101010u) == 0b111);
}

void test_expand_bitsr()
{
    ASSERT_S(expand_bitsr<std::uint8_t>(0b000u, 0b101010u) == 0b000000);
    ASSERT_S(expand_bitsr<std::uint8_t>(0b101u, 0b101011u) == 0b001001);
    ASSERT_S(expand_bitsr<std::uint8_t>(0b101u, 0b101010u) == 0b100010);
    ASSERT_S(expand_bitsr<std::uint8_t>(0b111u, 0b101010u) == 0b101010);
}

constexpr int seed = 0x12345;
constexpr int default_fuzz_count = 1024 * 1024;
using rng_type = std::mt19937_64;

template <typename T, T (&Fun)(T), T (&Naive)(T), int FuzzCount = default_fuzz_count>
void naive_fuzz_1()
{
    rng_type rng { seed };
    std::uniform_int_distribution<T> d;

    for (int i = 0; i < FuzzCount; ++i) {
        const T x = d(rng);
        const T actual = Fun(x);
        const T naive = Naive(x);
        ASSERT(actual == naive);
    }
}

template <typename T, T (&Fun)(T, T), T (&Naive)(T, T), int FuzzCount = default_fuzz_count>
void naive_fuzz_2()
{
    rng_type rng { seed };
    std::uniform_int_distribution<T> d;

    for (int i = 0; i < FuzzCount; ++i) {
        const T x = d(rng), m = d(rng);
        const T actual = Fun(x, m);
        const T naive = Naive(x, m);
        ASSERT(actual == naive);
    }
}

// clang-format off
constexpr void (*tests[])() = {
    test_is_pow2_or_zero,
    test_log2_floor,
    test_log2_ceil,
    test_alternate01,
    test_bipp,
    test_reverse_bits,
    test_compress_bitsr<compress_bitsr>,
    test_compress_bitsr<compress_bitsr_naive>,
    test_expand_bitsr,
    
#ifndef __INTELLISENSE__
    naive_fuzz_1<std::uint8_t,  reverse_bits, reverse_bits_naive>,
    naive_fuzz_1<std::uint16_t, reverse_bits, reverse_bits_naive>,
    naive_fuzz_1<std::uint32_t, reverse_bits, reverse_bits_naive>,
    naive_fuzz_1<std::uint64_t, reverse_bits, reverse_bits_naive>,

    naive_fuzz_1<std::uint8_t,  bitwise_inclusive_prefix_parity, bitwise_inclusive_prefix_parity_naive>,
    naive_fuzz_1<std::uint16_t, bitwise_inclusive_prefix_parity, bitwise_inclusive_prefix_parity_naive>,
    naive_fuzz_1<std::uint32_t, bitwise_inclusive_prefix_parity, bitwise_inclusive_prefix_parity_naive>,
    naive_fuzz_1<std::uint64_t, bitwise_inclusive_prefix_parity, bitwise_inclusive_prefix_parity_naive>,

    naive_fuzz_2<std::uint8_t,  compress_bitsr, compress_bitsr_naive>,
    naive_fuzz_2<std::uint16_t, compress_bitsr, compress_bitsr_naive>,
    naive_fuzz_2<std::uint32_t, compress_bitsr, compress_bitsr_naive>,
    naive_fuzz_2<std::uint64_t, compress_bitsr, compress_bitsr_naive>,

    naive_fuzz_2<std::uint8_t,  expand_bitsr, expand_bitsr_naive>,
    naive_fuzz_2<std::uint16_t, expand_bitsr, expand_bitsr_naive>,
    naive_fuzz_2<std::uint32_t, expand_bitsr, expand_bitsr_naive>,
    naive_fuzz_2<std::uint64_t, expand_bitsr, expand_bitsr_naive>,
#endif
};
// clang-format on

} // namespace

int main()
{
    for (void (*test)() : tests) {
        test();
    }

    std::cout << ":)" << '\n';
}