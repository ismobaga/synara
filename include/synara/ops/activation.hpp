#pragma once

#include "synara/tensor/tensor.hpp"
#include "synara/autograd/no_grad.hpp"

namespace synara
{

    Tensor relu(const Tensor &a);
    Tensor leaky_relu(const Tensor &a, Tensor::value_type negative_slope = 0.01f);
    Tensor sigmoid(const Tensor &a);
    Tensor tanh(const Tensor &a);
    Tensor softmax(const Tensor &a, int dim = -1);
    Tensor gelu(const Tensor &a);

} // namespace synara