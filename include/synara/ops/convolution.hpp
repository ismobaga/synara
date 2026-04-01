#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  Size stride_h = 1,
                  Size stride_w = 1,
                  Size pad_h = 0,
                  Size pad_w = 0,
                  Size dilation_h = 1,
                  Size dilation_w = 1,
                  Size groups = 1);

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  const Tensor &bias,
                  Size stride_h = 1,
                  Size stride_w = 1,
                  Size pad_h = 0,
                  Size pad_w = 0,
                  Size dilation_h = 1,
                  Size dilation_w = 1,
                  Size groups = 1);

} // namespace synara
