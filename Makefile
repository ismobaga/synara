.PHONY: build clean test help

BUILD_DIR := build

help:
	@echo "Available targets:"
	@echo "  build       - Build the C++ project using CMake"
	@echo "  test        - Run tests with ctest"
	@echo "  clean       - Clean build directory and CMake cache"
	@echo "  help        - Show this help message"

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && cmake --build .

test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory and CMake cache"
