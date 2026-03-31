#pragma once

#include <stdexcept>
#include <string>

namespace synara {

class SynaraError : public std::runtime_error {
public:
    explicit SynaraError(const std::string& message)
        : std::runtime_error(message) {}
};

class ShapeError : public SynaraError {
public:
    explicit ShapeError(const std::string& message)
        : SynaraError(message) {}
};

class IndexError : public SynaraError {
public:
    explicit IndexError(const std::string& message)
        : SynaraError(message) {}
};

class ValueError : public SynaraError {
public:
    explicit ValueError(const std::string& message)
        : SynaraError(message) {}
};

} // namespace synara