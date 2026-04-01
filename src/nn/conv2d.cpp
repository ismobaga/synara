#include "synara/nn/conv2d.hpp"

#include <string>
#include <utility>
#include <vector>

#include "synara/core/error.hpp"
#include "synara/ops/convolution.hpp"

namespace synara
{

    Conv2d::Conv2d(Size in_channels,
                   Size out_channels,
                   Size kernel_h,
                   Size kernel_w,
                   Size stride_h,
                   Size stride_w,
                   Size pad_h,
                   Size pad_w,
                                     bool use_bias,
                                     Size dilation_h,
                                     Size dilation_w,
                                     Size groups)
        : in_channels_(in_channels),
          out_channels_(out_channels),
          kernel_h_(kernel_h),
          kernel_w_(kernel_w),
          stride_h_(stride_h),
          stride_w_(stride_w),
          pad_h_(pad_h),
          pad_w_(pad_w),
                    dilation_h_(dilation_h),
                    dilation_w_(dilation_w),
                    groups_(groups),
          use_bias_(use_bias),
          weight_(make_weight()),
          bias_(make_bias())
    {
        if (in_channels_ == 0 || out_channels_ == 0)
        {
            throw ValueError("Conv2d::Conv2d(): channel dimensions must be > 0.");
        }
        if (kernel_h_ == 0 || kernel_w_ == 0)
        {
            throw ValueError("Conv2d::Conv2d(): kernel dimensions must be > 0.");
        }
        if (stride_h_ == 0 || stride_w_ == 0)
        {
            throw ValueError("Conv2d::Conv2d(): stride must be >= 1.");
        }
        if (dilation_h_ == 0 || dilation_w_ == 0)
        {
            throw ValueError("Conv2d::Conv2d(): dilation must be >= 1.");
        }
        if (groups_ == 0)
        {
            throw ValueError("Conv2d::Conv2d(): groups must be >= 1.");
        }
        if (in_channels_ % groups_ != 0)
        {
            throw ValueError("Conv2d::Conv2d(): in_channels must be divisible by groups.");
        }
        if (out_channels_ % groups_ != 0)
        {
            throw ValueError("Conv2d::Conv2d(): out_channels must be divisible by groups.");
        }
    }

    Tensor Conv2d::forward(const Tensor &input)
    {
        if (input.rank() != 4)
        {
            throw ShapeError("Conv2d::forward(): input must be rank 4 (N, C, H, W).");
        }
        if (input.shape()[1] != in_channels_)
        {
            throw ShapeError("Conv2d::forward(): input channel dimension mismatch.");
        }

        if (use_bias_)
        {
            return conv2d(input,
                          weight_.tensor(),
                          bias_.tensor(),
                          stride_h_,
                          stride_w_,
                          pad_h_,
                          pad_w_,
                          dilation_h_,
                          dilation_w_,
                          groups_);
        }
        return conv2d(input,
                      weight_.tensor(),
                      stride_h_,
                      stride_w_,
                      pad_h_,
                      pad_w_,
                      dilation_h_,
                      dilation_w_,
                      groups_);
    }

    std::vector<Parameter *> Conv2d::parameters()
    {
        if (use_bias_)
        {
            return {&weight_, &bias_};
        }
        return {&weight_};
    }

    StateDict Conv2d::state_dict(const std::string &prefix) const
    {
        StateDict out;
        out.emplace(prefix + "weight", Tensor::from_vector(weight_.tensor().shape(), std::vector<Tensor::value_type>(weight_.tensor().data(), weight_.tensor().data() + weight_.tensor().numel()), false));
        if (use_bias_)
        {
            out.emplace(prefix + "bias", Tensor::from_vector(bias_.tensor().shape(), std::vector<Tensor::value_type>(bias_.tensor().data(), bias_.tensor().data() + bias_.tensor().numel()), false));
        }
        return out;
    }

    void Conv2d::load_state_dict(const StateDict &state, const std::string &prefix)
    {
        const std::string weight_key = prefix + "weight";
        const auto w_it = state.find(weight_key);
        if (w_it == state.end())
        {
            throw ValueError("Conv2d::load_state_dict(): missing key '" + weight_key + "'.");
        }
        if (w_it->second.shape() != weight_.tensor().shape())
        {
            throw ShapeError("Conv2d::load_state_dict(): shape mismatch for key '" + weight_key + "'.");
        }
        for (Size i = 0; i < weight_.tensor().numel(); ++i)
        {
            weight_.tensor().data()[i] = w_it->second.data()[i];
        }

        if (!use_bias_)
        {
            return;
        }

        const std::string bias_key = prefix + "bias";
        const auto b_it = state.find(bias_key);
        if (b_it == state.end())
        {
            throw ValueError("Conv2d::load_state_dict(): missing key '" + bias_key + "'.");
        }
        if (b_it->second.shape() != bias_.tensor().shape())
        {
            throw ShapeError("Conv2d::load_state_dict(): shape mismatch for key '" + bias_key + "'.");
        }
        for (Size i = 0; i < bias_.tensor().numel(); ++i)
        {
            bias_.tensor().data()[i] = b_it->second.data()[i];
        }
    }

    Parameter &Conv2d::weight() noexcept
    {
        return weight_;
    }

    const Parameter &Conv2d::weight() const noexcept
    {
        return weight_;
    }

    Parameter &Conv2d::bias() noexcept
    {
        return bias_;
    }

    const Parameter &Conv2d::bias() const noexcept
    {
        return bias_;
    }

    Size Conv2d::in_channels() const noexcept
    {
        return in_channels_;
    }

    Size Conv2d::out_channels() const noexcept
    {
        return out_channels_;
    }

    Size Conv2d::kernel_h() const noexcept
    {
        return kernel_h_;
    }

    Size Conv2d::kernel_w() const noexcept
    {
        return kernel_w_;
    }

    Size Conv2d::stride_h() const noexcept
    {
        return stride_h_;
    }

    Size Conv2d::stride_w() const noexcept
    {
        return stride_w_;
    }

    Size Conv2d::pad_h() const noexcept
    {
        return pad_h_;
    }

    Size Conv2d::pad_w() const noexcept
    {
        return pad_w_;
    }

    Size Conv2d::dilation_h() const noexcept
    {
        return dilation_h_;
    }

    Size Conv2d::dilation_w() const noexcept
    {
        return dilation_w_;
    }

    Size Conv2d::groups() const noexcept
    {
        return groups_;
    }

    bool Conv2d::has_bias() const noexcept
    {
        return use_bias_;
    }

    Parameter Conv2d::make_weight() const
    {
        const Size in_per_group = in_channels_ / groups_;
        Tensor tensor = Tensor::zeros(Shape({out_channels_, in_per_group, kernel_h_, kernel_w_}), true);
        for (Size co = 0; co < out_channels_; ++co)
        {
            for (Size ci = 0; ci < in_per_group; ++ci)
            {
                for (Size kh = 0; kh < kernel_h_; ++kh)
                {
                    for (Size kw = 0; kw < kernel_w_; ++kw)
                    {
                        const Size flat = ((co * in_per_group + ci) * kernel_h_ + kh) * kernel_w_ + kw;
                        const long long centered = static_cast<long long>(flat % 7) - 3;
                        tensor.at({co, ci, kh, kw}) = static_cast<Tensor::value_type>(0.05f * centered);
                    }
                }
            }
        }
        return Parameter(std::move(tensor));
    }

    Parameter Conv2d::make_bias() const
    {
        return Parameter(Tensor::zeros(Shape({out_channels_}), true));
    }

} // namespace synara
