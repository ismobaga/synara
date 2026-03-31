add_executable(test_shape test_shape.cpp)
target_link_libraries(test_shape PRIVATE synara_tensor)
add_test(NAME test_shape COMMAND test_shape)
