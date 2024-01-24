cmake_minimum_required(VERSION 3.22)
project(c++26-bit-permutations)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test ${HEADERS} test.cpp bit_permutations.hpp bit_permutations_naive.hpp)
target_compile_options(test PRIVATE -Wall -Wextra)