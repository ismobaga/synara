cmake_minimum_required(VERSION 3.20)

project(synara LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(synara_tensor
	src/tensor/shape.cpp
	src/tensor/storage.cpp
	src/tensor/strides.cpp
	src/tensor/tensor.cpp
)

add_library(synara::tensor ALIAS synara_tensor)

target_include_directories(synara_tensor
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(synara_tensor PRIVATE /W4)
else()
    target_compile_options(synara_tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(synara tools/synara.cpp)


target_link_libraries(synara PRIVATE synara_tensor)


add_executable(synara_tensor_basics examples/00_tensor_basics.cpp
)
target_link_libraries(synara_tensor_basics PRIVATE synara_tensor)


add_subdirectory(tests)