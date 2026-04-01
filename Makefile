.PHONY: build clean test scaffold_test help

BUILD_DIR := build

help:
	@echo "Available targets:"
	@echo "  build       - Build the C++ project using CMake"
	@echo "  test        - Run tests with ctest"
	@echo "  scaffold_test TEST_NAME=test_name - Create tests/test_name.cpp"
	@echo "  clean       - Clean build directory and CMake cache"
	@echo "  help        - Show this help message"

build:
	@cmake -S . -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR)

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

scaffold_test: build
	@TEST_NAME="$(TEST_NAME)" cmake --build $(BUILD_DIR) --target scaffold_test
	@echo "Re-run: make build && make test"

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory and CMake cache"
