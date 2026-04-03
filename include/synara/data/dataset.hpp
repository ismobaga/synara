#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "synara/tensor/slice.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    class TensorDataset
    {
    public:
        TensorDataset(Tensor inputs, Tensor targets);

        std::size_t size() const noexcept;

        std::pair<Tensor, Tensor> get(std::size_t index) const;

    private:
        Tensor inputs_;
        Tensor targets_;
    };

} // namespace synara
