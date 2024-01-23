# C++26 Bit Permutations

This project is a reference implementation for
[Pxxxx: Bit Permutations](https://eisenwave.github.io/cpp-proposals/bit-permutations.html)
(new proposal, not yet published).

This project provides a single-header reference implementation of the functions

- `reverse_bits`
- `next_bit_permutation`
- `prev_bit_permutation`
- `compress_bitr`
- `expand_bitsr`
- `compress_bitsl`
- `expand_bitsl`

There are also implementations of existing `<bit>` functions for the purpose of testing.
The standard library functions don't support `_BitInt` or 128-bit integers, so it was necessary
to circumvent them:

- `popcount`
- `countl_zero`
- `countl_one`
- `countr_zero`
- `countr_one`

All functions are located in namespace `cxx26bp`.

This implementation aims to provide the fastest possible library implementation for each of these
functions, using any possible hardware support.
This project is portable, and tries to support

- **Architectures:** x86, ARM
- **Compilers:** MSVC, GCC, Clang