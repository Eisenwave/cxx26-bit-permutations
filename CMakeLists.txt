cmake_minimum_required(VERSION 3.22)
project(c++26-bit-permutations)

set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bit_permutations_test  ${HEADERS} test.cpp bit_permutations.hpp bit_permutations_naive.hpp)
target_compile_features(bit_permutations_test PRIVATE cxx_std_23)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(llvm_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

target_compile_options(bit_permutations_test PRIVATE
  "$<${gcc_like_cxx}:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_cxx}:-W3>"
  "$<${llvm_cxx}:-stdlib=libc++>"
)
target_link_options(bit_permutations_test PRIVATE
   "$<${llvm_cxx}:-stdlib=libc++>"
)
enable_testing()
add_test(bit_permutations_test bit_permutations_test)


