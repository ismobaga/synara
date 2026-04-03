#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor sum(const Tensor &a);
    Tensor mean(const Tensor &a);
    Tensor sum(const Tensor &a, int dim, bool keepdim = false);
    Tensor mean(const Tensor &a, int dim, bool keepdim = false);
    Tensor max(const Tensor &a, int dim, bool keepdim = false);
    Tensor min(const Tensor &a, int dim, bool keepdim = false);
    Tensor argmax(const Tensor &a, int dim, bool keepdim = false);
    Tensor argmin(const Tensor &a, int dim, bool keepdim = false);

} // namespace synara