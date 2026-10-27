# Build (from the cpp folder):
#   cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(licenta_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The engine itself, shared by the programs and the tests.
add_library(engine
    src/tokenizer.cpp
    src/inverted_index.cpp
    src/bm25scorer.cpp)
target_include_directories(engine PUBLIC include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(engine PRIVATE -Wall -Wextra)
endif()
# MinGW: put the C++ runtime inside each .exe. Otherwise Windows may load another program's
# libstdc++ DLL from PATH and the program stops at start (exit code 0xc0000139).
if(MINGW)
    add_link_options(-static)
endif()

add_executable(search_cli src/search_cli.cpp)
target_link_libraries(search_cli engine)

add_executable(demo src/main.cpp)
target_link_libraries(demo engine)

enable_testing()
foreach(name test_engine-v1 test_phrase test_scoring test_index_io)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} engine)
    add_test(NAME ${name} COMMAND ${name})
endforeach()
