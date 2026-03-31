#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "synara/tensor/shape.hpp"

namespace synara
{

    class Strides
    {
    public:
        Strides() = default;
        explicit Strides(std::vector<std::size_t> values);

        static Strides contiguous(const Shape &shape);

        const std::vector<std::size_t> &values() const noexcept;
        std::size_t rank() const noexcept;

        std::size_t operator[](std::size_t dim) const;

        bool is_contiguous(const Shape &shape) const;
        std::string to_string() const;

    private:
        std::vector<std::size_t> values_;
    };

} // namespace synara
