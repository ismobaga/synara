#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    double accuracy(const Tensor &predictions, const Tensor &targets);
    double binary_accuracy(const Tensor &predictions, const Tensor &targets, Tensor::value_type threshold = 0.5f);

} // namespace synara
