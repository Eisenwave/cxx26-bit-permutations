#ifndef CXX26_BIT_PERMUTATIONS_INCLUDE_GUARD
#define CXX26_BIT_PERMUTATIONS_INCLUDE_GUARD

#include <version>

// DETECT COMPILER

#ifdef __GNUC__
#define CXX26_BIT_PERMUTATIONS_GNU
#endif
#ifdef __clang__
#define CXX26_BIT_PERMUTATIONS_CLANG
#endif
#ifdef _MSC_VER
#define CXX26_BIT_PERMUTATIONS_MSVC
#endif

// DETECT ARCHITECTURE

#if defined(__x86_64__) || defined(_M_X64)
#define CXX26_BIT_PERMUTATIONS_X86_64
#define CXX26_BIT_PERMUTATIONS_X86
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#define CXX26_BIT_PERMUTATIONS_X86
#endif

#if defined(_M_ARM) || defined(__arm__)
#define CXX26_BIT_PERMUTATIONS_ARM
#endif

// DETECT INSTRUCTION SET FEATURES

#ifdef __BMI2__
#define CXX26_BIT_PERMUTATIONS_X86_BMI2
#endif
#ifdef __PCLMUL__
#define CXX26_BIT_PERMUTATIONS_X86_PCLMUL
#endif
#ifdef __POPCNT__
#define CXX26_BIT_PERMUTATIONS_X86_POPCNT
#endif
#ifdef __ARM_FEATURE_SVE2
#define CXX26_BIT_PERMUTATIONS_ARM_SVE2
#endif
#ifdef __ARM_FEATURE_SVE
#define CXX26_BIT_PERMUTATIONS_ARM_SVE
#endif
#ifdef __ARM_FEATURE_SME
#define CXX26_BIT_PERMUTATIONS_ARM_SME
#endif

// DEFINE INSTRUCTION SUPPORT BASED ON INSTRUCTION
// SET FEATURES

#if defined(CXX26_BIT_PERMUTATIONS_X86_BMI2)
#define CXX26_BIT_PERMUTATIONS_X86_PDEP
#define CXX26_BIT_PERMUTATIONS_X86_PEXT
#endif

#if defined(__ARM_FEATURE_SVE2)
#define CXX26_BIT_PERMUTATIONS_ARM_BDEP
#define CXX26_BIT_PERMUTATIONS_ARM_BEXT
#define CXX26_BIT_PERMUTATIONS_ARM_BGRP
#endif

#if defined(CXX26_BIT_PERMUTATIONS_ARM_SME) || defined(__ARM_FEATURE_SVE)
#define CXX26_BIT_PERMUTATIONS_ARM_RBIT
#endif

#if defined(CXX26_BIT_PERMUTATIONS_ARM_SME) || defined(__ARM_FEATURE_SVE)
// support for 64-bit PMUL is optional, so we will
// only use up to the 32-bit variant of this
#define CXX26_BIT_PERMUTATIONS_ARM_PMUL
#endif

// DEFINE WHICH FUNCTIONS HAVE "FAST" SUPPORT

#if defined(CXX26_BIT_PERMUTATIONS_ARM_RBIT)
#define CXX26_BIT_PERMUTATIONS_FAST_REVERSE
#endif

#if defined(CXX26_BIT_PERMUTATIONS_X86_PEXT) || defined(CXX26_BIT_PERMUTATIONS_ARM_BEXT)
#define CXX26_BIT_PERMUTATIONS_FAST_COMPRESS
#endif

#if defined(CXX26_BIT_PERMUTATIONS_X86_PDEP) || defined(CXX26_BIT_PERMUTATIONS_ARM_BDEP)
#define CXX26_BIT_PERMUTATIONS_FAST_EXPAND
#endif

#if defined(CXX26_BIT_PERMUTATIONS_X86_POPCNT) || defined(CXX26_BIT_PERMUTATIONS_X86_BMI2)         \
    || defined(CXX26_BIT_PERMUTATIONS_ARM_SVE)
#define CXX26_BIT_PERMUTATIONS_FAST_POPCOUNT
#endif

// INCLUDES

#ifdef CXX26_BIT_PERMUTATIONS_X86
#include <immintrin.h>
#endif
#ifdef CXX26_BIT_PERMUTATIONS_ARM_RBIT
#include <arm_acle.h>
#endif
#ifdef CXX26_BIT_PERMUTATIONS_ARM_PMUL
#include <arm_neon.h>
#endif
#ifdef CXX26_BIT_PERMUTATIONS_ARM_SVE2
#include <arm_sve.h>
#endif

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>

namespace cxx26bp {

namespace detail {

template <typename T>
inline constexpr int digits_v = std::numeric_limits<T>::digits;

template <typename T>
concept bit_uint =
#ifdef CXX26_BIT_PERMUTATIONS_CLANG
    requires(T x) { []<int N>(unsigned _BitInt(N)) {}(x); };

template <int N>
inline constexpr int digits_v<_BitInt(N)> = N;

static_assert(digits_v<_BitInt(128)> == 128);
#else
    false;
#endif

#ifdef CXX26_BIT_PERMUTATIONS_GNU
#define CXX26_BIT_PERMUTATIONS_U128
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using uint128_t = unsigned __int128;
#pragma GCC diagnostic pop
static_assert(std::numeric_limits<uint128_t>::digits == 128);
#else
struct uint128_t;
#endif

template <typename T>
concept permissive_unsigned_integral
    = std::unsigned_integral<T> || bit_uint<T> || std::same_as<T, uint128_t>;

/// Simpler form of `has_single_bit()` which doesn't complain about `int`.
[[nodiscard]] constexpr int is_pow2_or_zero(int x) noexcept
{
    return (x & (x - 1)) == 0;
}

/// @brief Creates a number with alternating
/// groups of 0s and 1s. The least significant bit
/// is always zero.
template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T alternate01(int group_size = 1) noexcept
{
    constexpr int N = digits_v<T>;
    constexpr T one = 1;

    if (group_size == 0 || group_size >= N) {
        return 0;
    }

    T result = ((one << group_size) - one) << group_size;

    for (int i = group_size << 1; i < N; i <<= 1) {
        result |= result << i;
    }

    return result;
}

// Exposed as a separate function for testing purposes.
template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T reverse_bits_naive(T x) noexcept
{
    constexpr int N = digits_v<T>;

    // Naive fallback.
    // O(N)
    T result = 0;
    for (int i = 0; i < N; ++i) {
        result <<= 1;
        result |= x & 1;
        x >>= 1;
    }
    return result;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countr_one_naive(T x) noexcept
{
    int result = 0;
    while (x & 1) {
        result++;
        x >>= 1;
    }
    return result;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countr_zero_naive(T x) noexcept
{
    return countr_one_naive(static_cast<T>(~x));
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countl_zero_naive(T x) noexcept
{
    return countr_zero_naive(reverse_bits_naive(x));
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countr_zero(T x) noexcept
{
    constexpr int N = digits_v<T>;

#ifdef CXX26_BIT_PERMUTATIONS_GNU
    constexpr int N_ull = digits_v<unsigned long long>;
    if constexpr (N <= N_ull) {
        if (x == 0) {
            return N;
        }
        if constexpr (N <= digits_v<unsigned>) {
            constexpr auto sentinel = (1u << (N - 1) << 1);
            return __builtin_ctz(x | sentinel);
        }
        else if constexpr (N <= digits_v<unsigned long>) {
            constexpr auto sentinel = (1ul << (N - 1) << 1);
            return __builtin_ctzl(x | sentinel);
        }
        else if constexpr (N <= digits_v<unsigned long long>) {
            constexpr auto sentinel = (1ull << (N - 1) << 1);
            return __builtin_ctzll(x | sentinel);
        }
    }
#endif
    // TODO: MSVC

    // https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
    constexpr int start = std::bit_ceil<unsigned>(N);

    int result = N;
    x &= -x; // isolate the lowest 1-bit
    result -= (x != 0);
    for (int i = start; i >>= 1;) {
        const T mask = static_cast<T>(~alternate01<T>(i));
        result -= ((x & mask) != 0) * i;
    }
    return result;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countr_one(T x) noexcept
{
    return countr_zero(static_cast<T>(~x));
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countl_zero(T x) noexcept
{
    constexpr int N = digits_v<T>;

    if (x == 0) {
        return N;
    }

#ifdef CXX26_BIT_PERMUTATIONS_GNU
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating countl_zero  =>  __builtin_clz
#endif
    if constexpr (N <= digits_v<unsigned>) {
        return __builtin_clz(x) - (digits_v<unsigned> - N);
    }
    else if constexpr (N <= digits_v<unsigned long>) {
        return __builtin_clzl(x) - (digits_v<unsigned long> - N);
    }
    else if constexpr (N <= digits_v<unsigned long long>) {
        return __builtin_clzll(x) - (digits_v<unsigned long long> - N);
    }
#endif
#ifdef CXX26_BIT_PERMUTATIONS_MSVC
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating countl_zero  =>  __lzcnt
#endif
    if !consteval {
        if constexpr (N <= 16) {
            return static_cast<int>(__lzcnt16(x)) - (16 - N);
        }
        else if constexpr (N <= 32) {
            return static_cast<int>(__lzcnt(x | sentinel)) - (32 - N);
        }
        else if constexpr (N <= 64) {
            return static_cast<int>(__lzcnt64(x | sentinel)) - (64 - N);
        }
    }
#endif
    constexpr int start = std::bit_ceil<unsigned>(N);
    auto base_mask = static_cast<T>(-T { 1 });
    int result = 0;
    for (int i = start; i >>= 1;) {
        base_mask <<= i;
        if ((x & (base_mask >> result)) == 0) {
            result += i;
        }
    }
    return result - (start - N);
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countl_one(T x) noexcept
{
    return countl_zero(static_cast<T>(~x));
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countl_one_naive(T x) noexcept
{
    return countl_zero_naive(static_cast<T>(~x));
}

/// Computes `floor(log2(max(1, x)))` of an
/// integer x.
[[nodiscard]] constexpr int log2_floor(int x) noexcept
{
    return x < 1 ? 0 : digits_v<unsigned> - countl_zero(static_cast<unsigned>(x)) - 1;
}

/// Computes `ceil(log2(max(1, x)))` of an
/// integer x.
[[nodiscard]] constexpr int log2_ceil(int x) noexcept
{
    return log2_floor(x) + !is_pow2_or_zero(x);
}

// `std::popcount` does not accept _BitInt or other extensions, so we make our own.
template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int popcount(T x) noexcept
{
    constexpr int N = digits_v<T>;

#ifdef CXX26_BIT_PERMUTATIONS_GNU
    if constexpr (N <= digits_v<unsigned>) {
        return __builtin_popcount(x);
    }
    else if constexpr (N <= digits_v<unsigned long>) {
        return __builtin_popcountl(x);
    }
    else if constexpr (N <= digits_v<unsigned long long>) {
        return __builtin_popcountll(x);
    }
#endif
#ifdef CXX26_BIT_PERMUTATIONS_MSVC
    if !consteval {
        if constexpr (N <= digits_v<unsigned short>) {
            return static_cast<int>(__popcnt16(x));
        }
        else if constexpr (N <= digits_v<unsigned int>) {
            return static_cast<int>(__popcnt(x));
        }
        else if constexpr (N <= 64) {
            return static_cast<int>(__popcnt64(x));
        }
    }
#endif
    constexpr int N_native = digits_v<size_t>;
    if constexpr (N > N_native) {
        int sum = 0;
        for (int i = 0; i < N; i += N_native) {
            sum += popcount(static_cast<size_t>(x));
            x >>= N_native;
        }
        return sum;
    }
    else if constexpr (N == 1) {
        return x;
    }
    else if constexpr (N == 2) {
        return (x >> 1) + (x & 1);
    }
    else if constexpr (N == 3) {
        return (x >> 2) + ((x >> 1) & 1) + (x & 1);
    }
    else {
        constexpr auto mask1 = static_cast<T>(~alternate01<T>(1));
        constexpr auto mask2 = static_cast<T>(~alternate01<T>(2));

        // TODO: investigate whether this really works for non-power-of-two integers
        //       I suspected that it does.
        T result = x - ((x >> 1) & mask1);
        result = ((result >> 2) & mask2) + (result & mask2);

        for (int i = 4; i < N; i <<= 1) {
            const auto mask = static_cast<T>(~alternate01<T>(i));
            result = ((result >> i) + result) & mask;
        }
        return result;
    }
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int popcount_naive(T x) noexcept
{
    int result = 0;
    while (x != 0) {
        result += x & 1;
        x >>= 1;
    }
    return result;
}

/// Each bit in `x` is converted to the parity a bit and all bits to its right.
/// This can also be expressed as `CLMUL(x, -1)` where `CLMUL` is a carry-less
/// multiplication.
template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T bitwise_inclusive_right_parity(T x) noexcept
{
    constexpr int N = digits_v<T>;

#ifdef CXX26_BIT_PERMUTATIONS_X86_PCLMUL
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating bitwise_inclusive_right_parity  =>  PCLMUL
#endif
    if !consteval {
        if constexpr (N <= 64) {
            const __m128i x_128 = _mm_set_epi64x(0, x);
            const __m128i neg1_128 = _mm_set_epi64x(0, -1);
            const __m128i result_128 = _mm_clmulepi64_si128(x_128, neg1_128, 0);
            return static_cast<T>(_mm_extract_epi64(result_128, 0));
        }
    }
#endif
    // TODO: Technically, ARM does have some support for polynomial multiplication prior to
    // SVE2.
    //       However, this support is fairly limited and it's not even clear whether it
    //       beats this implementation.
    for (int i = 1; i < N; i <<= 1) {
        x ^= x << i;
    }
    return x;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T bitwise_inclusive_right_parity_naive(T x) noexcept
{
    constexpr int N = digits_v<T>;

    T result = 0;
    bool parity = false;
    for (int i = 0; i < N; ++i) {
        parity ^= (x >> i) & 1;
        result |= static_cast<T>(parity) << i;
    }

    return result;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T next_bit_permutation_naive(T x) noexcept
{
    const int count = popcount(x);
    while (x != 0 && popcount(++x) != count) { }
    return x;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T prev_bit_permutation_naive(T x) noexcept
{
    const int count = popcount(x);
    while (x != 0 && popcount(--x) != count) { }
    return x;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T compress_bitsr_naive(T x, T m) noexcept
{
    constexpr int N = digits_v<T>;

    T result = 0;
    for (int i = 0, j = 0; i != N; ++i) {
        const bool mask_bit = (m >> i) & 1;
        result |= (mask_bit & (x >> i)) << j;
        j += mask_bit;
    }
    return result;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T compress_bitsl_naive(T x, T m) noexcept
{
    const T xr = reverse_bits_naive(x);
    const T mr = reverse_bits_naive(m);
    return reverse_bits_naive(compress_bitsr_naive(xr, mr));
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr T expand_bitsr_naive(T x, T m) noexcept
{
    constexpr int N = digits_v<T>;

    T result = 0;
    for (int i = 0, j = 0; i != N; ++i) {
        const bool mask_bit = (m >> i) & 1;
        result |= (mask_bit & (x >> j)) << i;
        j += mask_bit;
    }
    return result;
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T expand_bitsl_naive(T x, T m) noexcept
{
    const T xr = reverse_bits_naive(x);
    const T mr = reverse_bits_naive(m);
    return reverse_bits_naive(expand_bitsr_naive(xr, mr));
}

} // namespace detail

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T reverse_bits(T x) noexcept
{
    constexpr int N = detail::digits_v<T>;

// TODO: for integers >= 256-bit, it may be better to recursively split them into two halves until
//       the 64-bit version is available, and reassemble.
//       This is O(n) though and might be worse, if 256-bit shifts are available.
//       The current strategy for >= 256-bit is to rely on std::byteswap to do most of the job, and
//       then finish it with three more shifts.
#ifdef CXX26_BIT_PERMUTATIONS_CLANG
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating reverse_bits  =>  __builtin_bitreverse (clang)
#endif
    if constexpr (N <= 8) {
        return static_cast<T>(__builtin_bitreverse8(x) >> (8 - N));
    }
    else if constexpr (N == 16) {
        return __builtin_bitreverse16(x);
    }
    else if constexpr (N == 32) {
        return __builtin_bitreverse32(x);
    }
    else if constexpr (N == 64) {
        return __builtin_bitreverse64(x);
    }
#elif defined(CXX26_BIT_PERMUTATIONS_ARM_RBIT)
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating reverse_bits  =>  __rbit
#endif
    if !consteval {
        constexpr int N_uint = detail::digits_v<unsigned>;
        if constexpr (N <= N_uint) {
            return static_cast<T>(__rbit(static_cast<unsigned>(x)) >> (N_uint - N));
        }
        else if constexpr (N == detail::digits_v<unsigned long>) {
            return static_cast<T>(__rbitl(x));
        }
        else if constexpr (N == detail::digits_v<unsigned long long>) {
            return static_cast<T>(__rbitll(x));
        }
    }
#endif
    constexpr int N_native = detail::digits_v<size_t>;
    if constexpr (N > N_native && N % N_native == 0) {
        // For multiples of the native size, we assume that there is a fast native
        // implementation. We perform the naive algorithm, but for N_native bits at time, not
        // just one.
        T result = 0;
        for (int i = 0; i < N; i += N_native) {
            result <<= N_native;
            result |= reverse_bits(static_cast<size_t>(x));
            x >>= N_native;
        }
        return result;
    }
    else if constexpr (detail::is_pow2_or_zero(N)) {
        // Byte-swap and parallel swap technique for conventional architectures.
        // O(log N)
        constexpr int byte_bits = detail::digits_v<unsigned char>;
        int start_i = N;

        // If byteswap does what we want, we can skip a few iterations of the subsequent loop.
        if constexpr (detail::is_pow2_or_zero(byte_bits) && N >= byte_bits
                      && std::unsigned_integral<T>) {
            // TODO: implement detail::byteswap so that we can keep using this
            x = std::byteswap(x);
            start_i = byte_bits;
        }

        for (int i = start_i >> 1; i != 0; i >>= 1) {
            const T hi = detail::alternate01<T>(i);
            x = ((x & hi) >> i) | ((x & ~hi) << i);
        }

        return x;
    }
    else {
#ifdef CXX26_BIT_PERMUTATIONS_CLANG
        constexpr int M = std::bit_ceil<unsigned>(N);
        static_assert(M != N);
        return reverse_bits(static_cast<_BitInt(M)>(x)) >> (M - N);
#else
        return detail::reverse_bits_naive(x);
#endif
    }
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T next_bit_permutation(T x) noexcept
{
    // https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    constexpr T one = 1;
    const T t = x | (x - one);
    if (t == static_cast<T>(-1)) {
        return 0;
    }
    // Two shifts are better than shifting by + 1. We must not shift by the operand size.
    return (t + one) | (((~t & -~t) - one) >> detail::countr_zero(x) >> one);
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T prev_bit_permutation(T x) noexcept
{
    constexpr T one = 1;
    const T trailing_ones_cleared = x & (x + one);
    if (trailing_ones_cleared == 0) {
        return 0;
    }
    const T trailing_ones = x ^ trailing_ones_cleared;
    const int shift
        = detail::countr_zero(trailing_ones_cleared) - detail::countr_one(trailing_ones) - 1;

    return static_cast<T>(trailing_ones_cleared - one) >> shift << shift;
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T compress_bitsr(T x, T m) noexcept
{
    constexpr int N = detail::digits_v<T>;

#ifdef CXX26_BIT_PERMUTATIONS_X86_PEXT
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating compress_bitsr  =>  PEXT
#endif
    if !consteval {
        if constexpr (N <= 32) {
            return static_cast<T>(_pext_u32(x, m));
        }
        else if constexpr (N <= 64) {
            return static_cast<T>(_pext_u64(x, m));
        }
    }
#endif

#ifdef CXX26_BIT_PERMUTATIONS_ARM_BEXT
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating compress_bitsr  =>  BEXT
#endif
    if !consteval {
        if constexpr (N <= 8) {
            auto sv_result = svbext_u8(svdup_u8(x), svdup_u8(m));
            return static_cast<T>(svorv_u8(svptrue_b8(), sv_result));
        }
        else if constexpr (N <= 16) {
            auto sv_result = svbext_u16(svdup_u16(x), svdup_u16(m));
            return static_cast<T>(svorv_u16(svptrue_b16(), sv_result));
        }
        else if constexpr (N <= 32) {
            auto sv_result = svbext_u32(svdup_u32(x), svdup_u32(m));
            return static_cast<T>(svorv_u32(svptrue_b32(), sv_result));
        }
        else if constexpr (N <= 64) {
            auto sv_result = svbext_u64(svdup_u64(x), svdup_u64(m));
            return static_cast<T>(svorv_u64(svptrue_b64(), sv_result));
        }
    }
#endif
    constexpr int N_native = detail::digits_v<size_t>;
    if constexpr (N > N_native) {
        // For integer sizes above the native size, we assume that a fast native implementation
        // is provided. We then perform the algorithm digit by digit, where a digit is a native
        // integer.
        T result = 0;
        int offset = 0;
        for (int mask_bits = 0; mask_bits < N; mask_bits += N_native) {
            const auto compressed = compress_bitsr(static_cast<size_t>(x), static_cast<size_t>(m));
            result |= static_cast<T>(compressed) << offset;
            offset += detail::popcount(static_cast<size_t>(m));
            x >>= N_native;
            m >>= N_native;
        }

        return result;
    }
    else {
        x &= m;
        T mk = ~m << 1;

        for (int i = 1; i < N; i <<= 1) {
            const T mk_parity = detail::bitwise_inclusive_right_parity(mk);

            const T move = mk_parity & m;
            m = (m ^ move) | (move >> i);

            const T t = x & move;
            x = (x ^ t) | (t >> i);

            mk &= ~mk_parity;
        }
        return x;
    }
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T expand_bitsr(T x, T m) noexcept
{
    constexpr int N = detail::digits_v<T>;
    constexpr int log_N = detail::log2_floor(N);

#ifdef CXX26_BIT_PERMUTATIONS_X86_PDEP
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating expand_bitsr  =>  PDEP
#endif
    if !consteval {
        if constexpr (N <= 32) {
            return _pdep_u32(x, m);
        }
        else if constexpr (N <= 64) {
            return _pdep_u64(x, m);
        }
        // TODO 128-bit
    }
#endif

#ifdef CXX26_BIT_PERMUTATIONS_ARM_BDEP
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating expand_bitsr  =>  BDEP
#endif
    if !consteval {
        if constexpr (N <= 8) {
            auto sv_result = svbdep_u8(svdup_u8(x), svdup_u8(m));
            return static_cast<T>(svorv_u8(svptrue_b8(), sv_result));
        }
        else if constexpr (N <= 16) {
            auto sv_result = svbdep_u16(svdup_u16(x), svdup_u16(m));
            return static_cast<T>(svorv_u16(svptrue_b16(), sv_result));
        }
        else if constexpr (N <= 32) {
            auto sv_result = svbdep_u32(svdup_u32(x), svdup_u32(m));
            return static_cast<T>(svorv_u32(svptrue_b32(), sv_result));
        }
        else if constexpr (N <= 64) {
            auto sv_result = svbdep_u64(svdup_u64(x), svdup_u64(m));
            return static_cast<T>(svorv_u64(svptrue_b64(), sv_result));
        }
    }
#endif
    constexpr int N_native = detail::digits_v<size_t>;
    if constexpr (N > N_native) {
        // Digit-by-digit approach, same as in expand_bitsr.
        T result = 0;
        for (int mask_bits = 0; mask_bits < N; mask_bits += N_native) {
            const auto expanded = expand_bitsr(static_cast<size_t>(x), static_cast<size_t>(m));
            result |= static_cast<T>(expanded) << mask_bits;
            x >>= detail::popcount(static_cast<size_t>(m));
            m >>= N_native;
        }

        return result;
    }
    else {
        const T initial_m = m;

        T array[log_N];
        T mk = ~m << 1; // We will count 0's to right.

        for (int i = 0; i < log_N; ++i) {
            const T mk_parity = detail::bitwise_inclusive_right_parity(mk);
            const T move = mk_parity & m;
            m = (m ^ move) | (move >> (1 << i));
            array[i] = move;
            mk &= ~mk_parity;
        }

        for (int i = log_N; i-- > 0;) {
            const T move = array[i];
            const T t = x << (1 << i);
            x = (x & ~move) | (t & move);
        }

        return x & initial_m;
    }
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T compress_bitsl(T x, T m) noexcept
{
    constexpr int N = detail::digits_v<T>;

#if defined(CXX26_BIT_PERMUTATIONS_FAST_REVERSE) && !defined(CXX26_BIT_PERMUTATIONS_FAST_POPCOUNT)
    return reverse_bits(compress_bitsr(reverse_bits(x)), reverse_bits(m)));
#else
    if (m == 0) { // Prevents shift which is >= the operand size.
        return 0;
    }
    int shift = N - detail::popcount(m);
    return static_cast<T>(compress_bitsr(x, m) << shift);
#endif
}

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T expand_bitsl(T x, T m) noexcept
{
    constexpr int N = detail::digits_v<T>;

#if defined(CXX26_BIT_PERMUTATIONS_FAST_REVERSE) && !defined(CXX26_BIT_PERMUTATIONS_FAST_POPCOUNT)
    return reverse_bits(expand_bitsr(reverse_bits(x)), reverse_bits(m)));
#else
    if (m == 0) {
        return 0;
    }
    const int shift = N - detail::popcount(m);
    return expand_bitsr(static_cast<T>(x >> shift), m);
#endif
}

} // namespace cxx26bp

#endif