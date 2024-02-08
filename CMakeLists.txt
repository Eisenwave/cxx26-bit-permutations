cmake_minimum_required(VERSION 3.22)
project(c++26-bit-permutations)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test ${HEADERS} test.cpp bit_permutations.hpp bit_permutations_naive.hpp)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(llvm_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

target_compile_options(test PRIVATE
  "$<${gcc_like_cxx}:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_cxx}:-W3>"
  "$<${llvm_cxx}:-stdlib=libc++>"
)
target_link_options(test PRIVATE
   "$<${llvm_cxx}:-stdlib=libc++>"
)


