#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara {

Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);

} // namespace synara