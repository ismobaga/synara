#include "synara/tensor/strides.hpp"

#include <sstream>
#include <stdexcept>

namespace synara
{

    Strides::Strides(std::vector<std::size_t> values) : values_(std::move(values)) {}

    Strides Strides::contiguous(const Shape &shape)
    {
        std::vector<std::size_t> values(shape.rank(), 1);
        if (shape.rank() == 0)
        {
            return Strides(values);
        }

        for (std::size_t i = shape.rank(); i > 0; --i)
        {
            const std::size_t dim = i - 1;
            if (dim + 1 < shape.rank())
            {
                values[dim] = values[dim + 1] * shape[dim + 1];
            }
        }
        return Strides(values);
    }

    const std::vector<std::size_t> &Strides::values() const noexcept { return values_; }

    std::size_t Strides::rank() const noexcept { return values_.size(); }

    std::size_t Strides::operator[](std::size_t dim) const
    {
        if (dim >= values_.size())
        {
            throw std::out_of_range("stride dimension out of range");
        }
        return values_[dim];
    }

    bool Strides::is_contiguous(const Shape &shape) const
    {
        if (rank() != shape.rank())
        {
            return false;
        }
        return values_ == contiguous(shape).values();
    }

    std::string Strides::to_string() const
    {
        std::ostringstream oss;
        oss << "(";
        for (std::size_t i = 0; i < values_.size(); ++i)
        {
            oss << values_[i];
            if (i + 1 < values_.size())
            {
                oss << ", ";
            }
        }
        oss << ")";
        return oss.str();
    }

} // namespace synara
