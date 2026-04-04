cmake_minimum_required(VERSION 3.20)

project(synara LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SYNARA_ENABLE_OPENMP "Enable OpenMP-threaded execution in Synara compute kernels" ON)

add_library(synara_tensor
	src/tensor/shape.cpp
	src/tensor/storage.cpp
	src/tensor/strides.cpp
	src/tensor/tensor.cpp
	src/nn/module.cpp
	src/nn/module_utils.cpp
	src/nn/linear.cpp
	src/nn/layer_norm.cpp
	src/nn/conv2d.cpp
	src/nn/batch_norm.cpp
	src/nn/dropout.cpp
	src/nn/sequential.cpp
	src/ops/elementwise.cpp
    src/ops/math.cpp
    src/ops/reduction.cpp
    src/ops/linalg.cpp
	src/ops/shape.cpp
	src/ops/convolution.cpp
	src/ops/pooling.cpp
    src/ops/activation.cpp
    src/ops/loss.cpp
    src/optim/sgd.cpp
    src/optim/adam.cpp
    src/optim/rmsprop.cpp
	src/optim/lr_scheduler.cpp
	src/optim/grad_clip.cpp
	src/metrics/metrics.cpp
	src/serialize/state_dict.cpp
	src/autograd/nodes.cpp
	src/autograd/engine.cpp
	src/autograd/no_grad.cpp
	src/data/dataset.cpp
	src/data/dataloader.cpp
)

add_library(synara::tensor ALIAS synara_tensor)

target_include_directories(synara_tensor
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (SYNARA_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(synara_tensor PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(synara_tensor PUBLIC SYNARA_USE_OPENMP=1)
        message(STATUS "Synara: OpenMP support enabled")
    else()
        message(STATUS "Synara: OpenMP requested but not found; building without threaded kernels")
    endif()
endif()

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

add_executable(synara_ops_basics
    examples/01_ops_basics.cpp
)
target_link_libraries(synara_ops_basics PRIVATE synara_tensor)


add_executable(synara_autograd_basics
    examples/02_autograd_basics.cpp
)
target_link_libraries(synara_autograd_basics PRIVATE synara_tensor)

add_executable(synara_linear_regression
	examples/03_linear_regression.cpp
)
target_link_libraries(synara_linear_regression PRIVATE synara_tensor)

add_executable(synara_xor_mlp
    examples/04_xor_mlp.cpp
)
target_link_libraries(synara_xor_mlp PRIVATE synara_tensor)

add_executable(synara_conv2d_basics
	examples/05_conv2d_basics.cpp
)
target_link_libraries(synara_conv2d_basics PRIVATE synara_tensor)

add_executable(synara_pooling_basics
	examples/06_pooling_basics.cpp
)
target_link_libraries(synara_pooling_basics PRIVATE synara_tensor)

add_executable(synara_avg_pooling_basics
	examples/07_avg_pooling_basics.cpp
)
target_link_libraries(synara_avg_pooling_basics PRIVATE synara_tensor)

add_executable(synara_conv2d_advanced
	examples/08_conv2d_advanced.cpp
)
target_link_libraries(synara_conv2d_advanced PRIVATE synara_tensor)

add_executable(synara_mnist_cnn
	examples/09_mnist_cnn.cpp
)
target_link_libraries(synara_mnist_cnn PRIVATE synara_tensor)

add_custom_target(scaffold_test
	COMMAND /bin/sh -c "${CMAKE_COMMAND} -DTEST_NAME=\"$$TEST_NAME\" -DTESTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests\" -P \"${CMAKE_CURRENT_SOURCE_DIR}/cmake/scaffold_test.cmake\""
	COMMENT "Scaffold a test file: TEST_NAME=test_example make scaffold_test"
)


add_subdirectory(tests)