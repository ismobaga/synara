#pragma once
#include "synara/tensor/tensor.hpp"
namespace synara
{
    Tensor exp(const Tensor &a);
    Tensor log(const Tensor &a);
    Tensor log2(const Tensor &a);
    Tensor sqrt(const Tensor &a);
    Tensor pow(const Tensor &a, Tensor::value_type exponent);
    Tensor abs(const Tensor &a);
    Tensor sign(const Tensor &a);
    Tensor clamp(const Tensor &a, Tensor::value_type min_val, Tensor::value_type max_val);
} // namespace synara