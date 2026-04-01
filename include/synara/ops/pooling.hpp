#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor max_pool2d(const Tensor &input,
                      Size kernel_h,
                      Size kernel_w,
                      Size stride_h = 1,
                      Size stride_w = 1,
                      Size pad_h = 0,
                      Size pad_w = 0);

    Tensor avg_pool2d(const Tensor &input,
                      Size kernel_h,
                      Size kernel_w,
                      Size stride_h = 1,
                      Size stride_w = 1,
                      Size pad_h = 0,
                      Size pad_w = 0);

} // namespace synara
