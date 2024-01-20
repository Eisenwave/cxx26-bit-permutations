# C++26 Bit Permutations

This project provides a single-header reference implementation for the functions

- `reverse_bits`
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