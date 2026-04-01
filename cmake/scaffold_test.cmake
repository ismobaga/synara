if(NOT DEFINED TEST_NAME OR TEST_NAME STREQUAL "")
    message(FATAL_ERROR "TEST_NAME is required. Example: make scaffold_test TEST_NAME=test_vector")
endif()

if(NOT TEST_NAME MATCHES "^test_[A-Za-z0-9_]+$")
    message(FATAL_ERROR "TEST_NAME must match ^test_[A-Za-z0-9_]+$")
endif()

if(NOT DEFINED TESTS_DIR OR TESTS_DIR STREQUAL "")
    message(FATAL_ERROR "TESTS_DIR is required")
endif()

set(TEST_FILE "${TESTS_DIR}/${TEST_NAME}.cpp")

if(EXISTS "${TEST_FILE}")
    message(STATUS "Test file already exists: ${TEST_FILE}")
    return()
endif()

file(WRITE "${TEST_FILE}" "#include <cassert>\n\nint main()\n{\n    // TODO: implement ${TEST_NAME}\n    assert(true);\n    return 0;\n}\n")

message(STATUS "Created ${TEST_FILE}")
message(STATUS "Reconfigure/build to register the new test:")
message(STATUS "  cmake -S . -B build && cmake --build build")
