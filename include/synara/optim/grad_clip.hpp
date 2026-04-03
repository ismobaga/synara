#pragma once

#include <vector>

#include "synara/tensor/tensor.hpp"

namespace synara
{

    double clip_grad_norm_(const std::vector<Tensor *> &parameters, double max_norm);
    void clip_grad_value_(const std::vector<Tensor *> &parameters, double max_value);

} // namespace synara
