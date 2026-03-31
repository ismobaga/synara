#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace synara
{

    class Shape
    {
    public:
        Shape() = default;
        explicit Shape(std::vector<std::size_t> dims);
        Shape(std::initializer_list<std::size_t> dims);

        const std::vector<std::size_t> &dims() const noexcept;
        std::size_t rank() const noexcept;
        std::size_t numel() const noexcept;

        std::size_t operator[](std::size_t dim) const;

        bool operator==(const Shape &other) const noexcept;
        bool operator!=(const Shape &other) const noexcept;

        std::string to_string() const;

    private:
        std::vector<std::size_t> dims_;
    };

} // namespace synara
