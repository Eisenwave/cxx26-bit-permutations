#ifndef CXX26_BIT_PERMUTATIONS_NAIVE_INCLUDE_GUARD
#define CXX26_BIT_PERMUTATIONS_NAIVE_INCLUDE_GUARD

#include "bit_permutations.hpp"

namespace cxx26bp::detail {

/// @brief Repeats a bit pattern.
/// @param x the bit-pattern, stored in the lest significant `length` bits.
/// @param length the length of the bit-pattern, in range [1, N]
/// @return The bit pattern in `x`, repeated as many times as representable by `T`.
/// @throws Nothing.
template <permissive_unsigned_integral T>
[[nodiscard]] CXX26_BIT_PERMUTATIONS_ALWAYS_INLINE constexpr T repeat_bits_naive(T x, int length)
{
    constexpr int N = digits_v<T>;

    T result = 0;
    for (int i = 0; i < N; ++i) {
        result |= ((x >> (i % length)) & 1) << i;
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
[[nodiscard]] constexpr int countl_one_naive(T x) noexcept
{
    return countl_zero_naive(static_cast<T>(~x));
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

} // namespace cxx26bp::detail

#endif