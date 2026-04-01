#pragma once

#include <utility>

#include "synara/tensor/tensor.hpp"

namespace synara
{

    class Parameter
    {
    public:
        Parameter() = default;
        explicit Parameter(Tensor value) : value_(std::move(value))
        {
            value_.set_requires_grad(true);
            value_.set_leaf(true);
        }

        Tensor &tensor() noexcept { return value_; }
        const Tensor &tensor() const noexcept { return value_; }

        bool requires_grad() const noexcept { return value_.requires_grad(); }
        bool has_grad() const noexcept { return value_.has_grad(); }

    private:
        Tensor value_;
    };

} // namespace synara
