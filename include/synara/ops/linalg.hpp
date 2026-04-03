#pragma once

#include "synara/tensor/tensor.hpp"
#include "synara/autograd/no_grad.hpp"

namespace synara
{

    Tensor matmul(const Tensor &a, const Tensor &b);
    Tensor embedding(const Tensor &indices, const Tensor &weight);

} // namespace synara
