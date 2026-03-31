#include "synara/tensor/shape.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace synara
{

    Shape::Shape(std::vector<std::size_t> dims) : dims_(std::move(dims)) {}

    Shape::Shape(std::initializer_list<std::size_t> dims) : dims_(dims) {}

    const std::vector<std::size_t> &Shape::dims() const noexcept { return dims_; }

    std::size_t Shape::rank() const noexcept { return dims_.size(); }

    std::size_t Shape::numel() const noexcept
    {
        if (dims_.empty())
        {
            return 1;
        }
        return std::accumulate(
            dims_.begin(), dims_.end(), static_cast<std::size_t>(1), std::multiplies<>());
    }

    std::size_t Shape::operator[](std::size_t dim) const
    {
        if (dim >= dims_.size())
        {
            throw std::out_of_range("shape dimension out of range");
        }
        return dims_[dim];
    }

    bool Shape::operator==(const Shape &other) const noexcept { return dims_ == other.dims_; }

    bool Shape::operator!=(const Shape &other) const noexcept { return !(*this == other); }

    std::string Shape::to_string() const
    {
        std::ostringstream oss;
        oss << "(";
        for (std::size_t i = 0; i < dims_.size(); ++i)
        {
            oss << dims_[i];
            if (i + 1 < dims_.size())
            {
                oss << ", ";
            }
        }
        oss << ")";
        return oss.str();
    }

} // namespace synara
