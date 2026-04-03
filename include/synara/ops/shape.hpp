#pragma once

#include <vector>

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor squeeze(const Tensor &a, int dim = -1);
    Tensor unsqueeze(const Tensor &a, int dim);
    Tensor permute(const Tensor &a, const std::vector<int> &dims);
    Tensor expand(const Tensor &a, const Shape &shape);
    Tensor broadcast_to(const Tensor &a, const Shape &shape);

    Tensor cat(const std::vector<Tensor> &tensors, int dim = 0);
    Tensor stack(const std::vector<Tensor> &tensors, int dim = 0);
    std::vector<Tensor> split(const Tensor &a, int split_size, int dim = 0);

} // namespace synara
