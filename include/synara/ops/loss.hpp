#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor mse_loss(const Tensor &pred, const Tensor &target);
    Tensor binary_cross_entropy(const Tensor &pred, const Tensor &target);

} // namespace synara