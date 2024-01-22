# C++26 Bit Permutations

This project is a reference implementation for
[Pxxxx: Bit Permutations](https://eisenwave.github.io/cpp-proposals/bit-permutations.html)
(new proposal, not yet published).

This project provides a single-header reference implementation for the functions

- `reverse_bits`
- `next_bit_permutation`
- `compress_bitr`
- `expand_bitsr`
- `compress_bitsl`
- `expand_bitsl`

All functions are located in namespace `std::experimental`.

This implementation aims to provide the fastest possible library implementation for each of these
functions, using any possible hardware support.
This project is portable, and tries to support

- **Architectures:** x86, ARM
- **Compilers:** MSVC, GCC, Clang