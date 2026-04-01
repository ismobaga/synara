#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor relu(const Tensor &a);
    Tensor sigmoid(const Tensor &a);

} // namespace synara