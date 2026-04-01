#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/pooling.hpp"

namespace synara
{

    class MaxPool2d : public Module
    {
    public:
        MaxPool2d(Size kernel_h,
                  Size kernel_w,
                  Size stride_h = 1,
                  Size stride_w = 1,
                  Size pad_h = 0,
                  Size pad_w = 0)
            : kernel_h_(kernel_h),
              kernel_w_(kernel_w),
              stride_h_(stride_h),
              stride_w_(stride_w),
              pad_h_(pad_h),
              pad_w_(pad_w)
        {
        }

        Tensor forward(const Tensor &input) override
        {
            return max_pool2d(input, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);
        }

        std::vector<Parameter *> parameters() override { return {}; }

    private:
        Size kernel_h_;
        Size kernel_w_;
        Size stride_h_;
        Size stride_w_;
        Size pad_h_;
        Size pad_w_;
    };

} // namespace synara
