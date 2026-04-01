#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  Size stride_h = 1,
                  Size stride_w = 1,
                  Size pad_h = 0,
                  Size pad_w = 0);

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  const Tensor &bias,
                  Size stride_h = 1,
                  Size stride_w = 1,
                  Size pad_h = 0,
                  Size pad_w = 0);

} // namespace synara
