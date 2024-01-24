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

#ifdef __BMI__
#define CXX26_BIT_PERMUTATIONS_X86_BMI
#endif
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

#ifdef CXX26_BIT_PERMUTATIONS_GNU
#define CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL _Pragma("GCC unroll 16")
#else
#define CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
#endif

namespace cxx26bp {

namespace detail {

template <typename T>
inline constexpr int digits_v = std::numeric_limits<T>::digits;

template <typename T>
struct is_bit_uint : std::false_type { };

#ifdef CXX26_BIT_PERMUTATIONS_CLANG
#define CXX26_BIT_PERMUTATIONS_BITINT
template <int N>
struct is_bit_uint<unsigned _BitInt(N)> : std::true_type { };

static_assert(is_bit_uint<unsigned _BitInt(2)>::value);
static_assert(is_bit_uint<unsigned _BitInt(8)>::value);

template <int N>
inline constexpr int digits_v<_BitInt(N)> = N - 1;

template <int N>
inline constexpr int digits_v<unsigned _BitInt(N)> = N;

static_assert(digits_v<_BitInt(128)> == 127);
static_assert(digits_v<unsigned _BitInt(128)> == 128);
#endif

template <typename T>
concept bit_uint = is_bit_uint<T>::value;

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

    CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
    for (int i = group_size << 1; i < N; i <<= 1) {
        result |= result << i;
    }

    return result;
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
#elif defined(CXX26_BIT_PERMUTATIONS_X86_BMI)
    if !consteval {
        if constexpr (N <= 16) {
            constexpr auto sentinel = static_cast<unsigned short>(1u << (N - 1) << 1);
            return _tzcnt_u16(x | sentinel);
        }
        else if constexpr (N <= 32) {
            constexpr auto sentinel = 1u << (N - 1) << 1;
            return _tzcnt_u32(x | sentinel);
        }
        else if constexpr (N <= 64) {
            constexpr auto sentinel = 1ull << (N - 1) << 1;
            return _tzcnt_u64(x | sentinel);
        }
    }
#elif defined(CXX26_BIT_PERMUTATIONS_X86)
    if !consteval {
        if constexpr (N <= 32) {
            constexpr unsigned __int32 sentinel = (1u << (N - 1) << 1);
            unsigned __int32 index;
            return _BitScanForward(&index, x | sentinel) ? static_cast<int>(index) : 0;
        }
        else if constexpr (N <= 64) {
            constexpr unsigned __int64 sentinel = (1ull << (N - 1) << 1);
            unsigned __int32 index;
            return _BitScanForward64(&index, x | sentinel) ? static_cast<int>(index) : 0;
        }
    }
#endif
    constexpr int N_nat = digits_v<std::size_t>;
    if constexpr (N > N_nat) {
        // Handle everything except for the most significant digit
        int result = 0;
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = 0; i + N_nat < N; i += N_nat) {
            const int part = countr_zero(static_cast<std::size_t>(x));
            result += part;
            if (part != N_nat) {
                return result;
            }
            x >>= N_nat;
        }
        // Handle the most significant digit.
        constexpr auto sentinel
            = static_cast<std::size_t>(N % N_nat == 0 ? 0 : std::size_t { 1 } << (N % N_nat));
        return result
            + countr_zero(static_cast<std::size_t>(static_cast<std::size_t>(x) | sentinel));
    }
    // https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
    else {
        int result = std::bit_ceil<unsigned>(N);
        x &= -x; // isolate the lowest 1-bit
        result -= (x != 0);
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = N; i >>= 1;) {
            const T mask = static_cast<T>(~alternate01<T>(i));
            result -= ((x & mask) != 0) * i;
        }
        if constexpr (is_pow2_or_zero(N)) {
            return result;
        }
        else {
            return result < N ? result : N;
        }
    }
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
            return static_cast<int>(__lzcnt(x)) - (32 - N);
        }
        else if constexpr (N <= 64) {
            return static_cast<int>(__lzcnt64(x)) - (64 - N);
        }
    }
#endif
    if (x == 0) {
        return N;
    }
    constexpr int start = std::bit_ceil<unsigned>(N);
    auto mask = static_cast<T>(-1);
    int result = 0;
    CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
    for (int i = start >> 1; i != 0; i >>= 1) {
        if (x & (mask << i)) {
            mask <<= i;
            result += i;
        }
    }
    return N - result - 1;
}

template <permissive_unsigned_integral T>
[[nodiscard]] constexpr int countl_one(T x) noexcept
{
    return countl_zero(static_cast<T>(~x));
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
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
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

        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = 4; i < N; i <<= 1) {
            const auto mask = static_cast<T>(~alternate01<T>(i));
            result = ((result >> i) + result) & mask;
        }
        return result;
    }
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

template <int N, permissive_unsigned_integral T>
[[nodiscard]] constexpr T reverse_bits_impl(T x) noexcept
{
    constexpr int N_actual = digits_v<T>;
    static_assert(N <= N_actual);

#ifdef CXX26_BIT_PERMUTATIONS_CLANG
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating reverse_bits  =>  __builtin_bitreverse (clang)
#endif
    if constexpr (N <= 8) {
        return static_cast<T>(__builtin_bitreverse8(x) >> (8 - N));
    }
    else if constexpr (N <= 16) {
        return static_cast<T>(__builtin_bitreverse16(x) >> (16 - N));
    }
    else if constexpr (N <= 32) {
        return static_cast<T>(__builtin_bitreverse32(x) >> (32 - N));
    }
    else if constexpr (N <= 64) {
        return static_cast<T>(__builtin_bitreverse64(x) >> (64 - N));
    }
#elif defined(CXX26_BIT_PERMUTATIONS_ARM_RBIT)
#ifdef CXX26_BIT_PERMUTATIONS_ENABLE_DEBUG_PP
#warning Delegating reverse_bits  =>  __rbit
#endif
    constexpr int N_ull = digits_v<unsigned long long>;
    if constexpr (N <= N_ull) {
        constexpr int N_u = digits_v<unsigned>;
        constexpr int N_ul = digits_v<unsigned long>;
        if !consteval {
            if constexpr (N <= N_u) {
                return static_cast<T>(__rbit(x) >> (N_u - N));
            }
            else if constexpr (N <= N_ul) {
                return static_cast<T>(__rbitl(x) >> (N_ul - N));
            }
            else if constexpr (N <= N_ull) {
                return static_cast<T>(__rbitll(x) >> (N_ull - N));
            }
        }
    }
#else
    if constexpr (false) { }
#endif
    else if constexpr (constexpr int N_native = digits_v<std::size_t>; N > N_native) {
        // N, rounded up to the next multiple of N_native.
        constexpr int N_ceil = N_native * (N / N_native + (N % N_native != 0));
        static_assert(N_ceil >= N);
        constexpr int shift = N_ceil - N;

        T most = 0;
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = 0; i + N_native < N; i += N_native) {
            most <<= N_native;
            most |= reverse_bits_impl<N_native>(static_cast<std::size_t>(x));
            x >>= N_native;
        }
        const T last = reverse_bits_impl<N_native>(static_cast<std::size_t>(x));

        return (most << (N_native - shift)) | (last >> shift);
    }
    else if constexpr (is_pow2_or_zero(N)) {
        // Byte-swap and parallel swap technique for conventional architectures.
        // O(log N)
        constexpr int byte_bits = digits_v<unsigned char>;
        int start_i = N;

        // If byteswap does what we want, we can skip a few iterations of the subsequent loop.
        if constexpr (detail::is_pow2_or_zero(byte_bits) && N >= byte_bits
                      && std::unsigned_integral<T>) {
            // TODO: implement detail::byteswap so that we can keep using this for _BitInt et al.
            x = std::byteswap(x) >> (N_actual - N);
            start_i = byte_bits;
        }

        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = start_i >> 1; i != 0; i >>= 1) {
            const T hi = alternate01<T>(i);
            x = ((x & hi) >> i) | ((x & ~hi) << i);
        }

        return x;
    }
    else {
        //         [hi] [lo]
        // input:  -654 3210
        // rev:    456- 0123
        // return: -0123 456

        constexpr int M = std::bit_floor<unsigned>(N);
        constexpr int shift = (M * 2) - N;
        static_assert(M < N);
        static_assert(M > shift);
        constexpr T lo_mask = (T { 1 } << M) - 1;

        const T lo = reverse_bits_impl<M>(x & lo_mask);
        const T hi = reverse_bits_impl<M>(x >> M);

        const T result = (lo << (M - shift)) | (hi >> shift);

        return result;
    }
}

} // namespace detail

template <detail::permissive_unsigned_integral T>
[[nodiscard]] constexpr T reverse_bits(T x) noexcept
{
    constexpr int N = detail::digits_v<T>;
    return detail::reverse_bits_impl<N>(x);
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
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
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

        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
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
    constexpr int log_N = detail::log2_floor(std::bit_ceil<unsigned>(N));

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
        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
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
        T mk = ~m << 1;

        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = 0; i < log_N; ++i) {
            const T mk_parity = detail::bitwise_inclusive_right_parity(mk);
            const T move = mk_parity & m;
            m = (m ^ move) | (move >> (1 << i));
            array[i] = move;
            mk &= ~mk_parity;
        }

        CXX26_BIT_PERMUTATIONS_AGGRESSIVE_UNROLL
        for (int i = log_N; i > 0;) {
            --i; // Normally, I would write (i-- > 0), but this triggers
                 // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113581
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