#include "synara/ops/convolution.hpp"

#include <memory>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        struct Conv2dConfig
        {
            Size stride_h;
            Size stride_w;
            Size pad_h;
            Size pad_w;
            Size dilation_h;
            Size dilation_w;
            Size groups;
        };

        bool valid_bias_shape(const Tensor &bias, Size out_channels)
        {
            if (bias.rank() == 1)
            {
                return bias.shape()[0] == out_channels;
            }
            if (bias.rank() == 2)
            {
                return bias.shape()[0] == 1 && bias.shape()[1] == out_channels;
            }
            return false;
        }

        Tensor::value_type bias_value(const Tensor &bias, Size c)
        {
            return (bias.rank() == 1) ? bias.at({c}) : bias.at({0, c});
        }

        class Conv2dNode : public Node
        {
        public:
            Conv2dNode(Tensor input,
                       Tensor weight,
                       Tensor bias,
                       bool use_bias,
                       Conv2dConfig cfg)
                : input_(std::move(input)),
                  weight_(std::move(weight)),
                  bias_(std::move(bias)),
                  use_bias_(use_bias),
                  cfg_(cfg)
            {
            }

            void backward(const Tensor &grad_output) override
            {
                const Size n = input_.shape()[0];
                const Size h_in = input_.shape()[2];
                const Size w_in = input_.shape()[3];

                const Size c_out = weight_.shape()[0];
                const Size c_in_per_group = weight_.shape()[1];
                const Size k_h = weight_.shape()[2];
                const Size k_w = weight_.shape()[3];
                const Size c_out_per_group = c_out / cfg_.groups;

                const Size h_out = grad_output.shape()[2];
                const Size w_out = grad_output.shape()[3];

                if (input_.requires_grad())
                {
                    Tensor grad_input = Tensor::zeros(input_.shape(), false);

                    for (Size b = 0; b < n; ++b)
                    {
                        for (Size co = 0; co < c_out; ++co)
                        {
                            const Size gidx = co / c_out_per_group;
                            const Size ci_start = gidx * c_in_per_group;
                            for (Size oh = 0; oh < h_out; ++oh)
                            {
                                for (Size ow = 0; ow < w_out; ++ow)
                                {
                                    const Tensor::value_type g = grad_output.at({b, co, oh, ow});
                                    for (Size ci_local = 0; ci_local < c_in_per_group; ++ci_local)
                                    {
                                        const Size ci = ci_start + ci_local;
                                        for (Size kh = 0; kh < k_h; ++kh)
                                        {
                                            const long long ih = static_cast<long long>(oh * cfg_.stride_h + kh * cfg_.dilation_h) - static_cast<long long>(cfg_.pad_h);
                                            if (ih < 0 || ih >= static_cast<long long>(h_in))
                                            {
                                                continue;
                                            }

                                            for (Size kw = 0; kw < k_w; ++kw)
                                            {
                                                const long long iw = static_cast<long long>(ow * cfg_.stride_w + kw * cfg_.dilation_w) - static_cast<long long>(cfg_.pad_w);
                                                if (iw < 0 || iw >= static_cast<long long>(w_in))
                                                {
                                                    continue;
                                                }

                                                grad_input.at({b, ci, static_cast<Size>(ih), static_cast<Size>(iw)}) +=
                                                    g * weight_.at({co, ci_local, kh, kw});
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    input_.accumulate_grad(grad_input);
                    if (input_.grad_fn())
                    {
                        input_.grad_fn()->backward(grad_input);
                    }
                }

                if (weight_.requires_grad())
                {
                    Tensor grad_weight = Tensor::zeros(weight_.shape(), false);

                    for (Size b = 0; b < n; ++b)
                    {
                        for (Size co = 0; co < c_out; ++co)
                        {
                            const Size gidx = co / c_out_per_group;
                            const Size ci_start = gidx * c_in_per_group;
                            for (Size oh = 0; oh < h_out; ++oh)
                            {
                                for (Size ow = 0; ow < w_out; ++ow)
                                {
                                    const Tensor::value_type g = grad_output.at({b, co, oh, ow});
                                    for (Size ci_local = 0; ci_local < c_in_per_group; ++ci_local)
                                    {
                                        const Size ci = ci_start + ci_local;
                                        for (Size kh = 0; kh < k_h; ++kh)
                                        {
                                            const long long ih = static_cast<long long>(oh * cfg_.stride_h + kh * cfg_.dilation_h) - static_cast<long long>(cfg_.pad_h);
                                            if (ih < 0 || ih >= static_cast<long long>(h_in))
                                            {
                                                continue;
                                            }

                                            for (Size kw = 0; kw < k_w; ++kw)
                                            {
                                                const long long iw = static_cast<long long>(ow * cfg_.stride_w + kw * cfg_.dilation_w) - static_cast<long long>(cfg_.pad_w);
                                                if (iw < 0 || iw >= static_cast<long long>(w_in))
                                                {
                                                    continue;
                                                }

                                                grad_weight.at({co, ci_local, kh, kw}) +=
                                                    g * input_.at({b, ci, static_cast<Size>(ih), static_cast<Size>(iw)});
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    weight_.accumulate_grad(grad_weight);
                    if (weight_.grad_fn())
                    {
                        weight_.grad_fn()->backward(grad_weight);
                    }
                }

                if (use_bias_ && bias_.requires_grad())
                {
                    Tensor grad_bias = Tensor::zeros(bias_.shape(), false);
                    for (Size b = 0; b < n; ++b)
                    {
                        for (Size co = 0; co < c_out; ++co)
                        {
                            Tensor::value_type acc = 0.0f;
                            for (Size oh = 0; oh < h_out; ++oh)
                            {
                                for (Size ow = 0; ow < w_out; ++ow)
                                {
                                    acc += grad_output.at({b, co, oh, ow});
                                }
                            }

                            if (grad_bias.rank() == 1)
                            {
                                grad_bias.at({co}) += acc;
                            }
                            else
                            {
                                grad_bias.at({0, co}) += acc;
                            }
                        }
                    }

                    bias_.accumulate_grad(grad_bias);
                    if (bias_.grad_fn())
                    {
                        bias_.grad_fn()->backward(grad_bias);
                    }
                }
            }

        private:
            Tensor input_;
            Tensor weight_;
            Tensor bias_;
            bool use_bias_;
            Conv2dConfig cfg_;
        };

        void validate_conv2d_shapes(const Tensor &input, const Tensor &weight, const Conv2dConfig &cfg)
        {
            if (input.rank() != 4)
            {
                throw ShapeError("conv2d(): input must be rank 4 (N, C_in, H, W).");
            }
            if (weight.rank() != 4)
            {
                throw ShapeError("conv2d(): weight must be rank 4 (C_out, C_in/groups, K_h, K_w).");
            }
            if (cfg.groups == 0)
            {
                throw ValueError("conv2d(): groups must be >= 1.");
            }

            const Size c_in = input.shape()[1];
            const Size c_out = weight.shape()[0];
            const Size c_in_per_group = weight.shape()[1];

            if (c_in_per_group == 0)
            {
                throw ShapeError("conv2d(): weight input channels per group must be > 0.");
            }
            if (c_in != c_in_per_group * cfg.groups)
            {
                throw ShapeError("conv2d(): input channels must match weight channels * groups.");
            }
            if (c_out % cfg.groups != 0)
            {
                throw ShapeError("conv2d(): output channels must be divisible by groups.");
            }
        }

        Tensor conv2d_impl(const Tensor &input,
                           const Tensor &weight,
                           const Tensor *bias,
                           Conv2dConfig cfg)
        {
            validate_conv2d_shapes(input, weight, cfg);

            if (cfg.stride_h == 0 || cfg.stride_w == 0)
            {
                throw ValueError("conv2d(): stride must be >= 1.");
            }
            if (cfg.dilation_h == 0 || cfg.dilation_w == 0)
            {
                throw ValueError("conv2d(): dilation must be >= 1.");
            }

            const Size n = input.shape()[0];
            const Size c_in = input.shape()[1];
            const Size h_in = input.shape()[2];
            const Size w_in = input.shape()[3];

            const Size c_out = weight.shape()[0];
            const Size c_in_per_group = weight.shape()[1];
            const Size k_h = weight.shape()[2];
            const Size k_w = weight.shape()[3];
            const Size c_out_per_group = c_out / cfg.groups;

            (void)c_in;

            const Size padded_h = h_in + 2 * cfg.pad_h;
            const Size padded_w = w_in + 2 * cfg.pad_w;
            const Size eff_kh = cfg.dilation_h * (k_h - 1) + 1;
            const Size eff_kw = cfg.dilation_w * (k_w - 1) + 1;

            if (k_h == 0 || k_w == 0)
            {
                throw ShapeError("conv2d(): kernel spatial dims must be > 0.");
            }
            if (padded_h < eff_kh || padded_w < eff_kw)
            {
                throw ShapeError("conv2d(): kernel larger than padded input.");
            }

            const Size h_out = (padded_h - eff_kh) / cfg.stride_h + 1;
            const Size w_out = (padded_w - eff_kw) / cfg.stride_w + 1;

            if (bias != nullptr && !valid_bias_shape(*bias, c_out))
            {
                throw ShapeError("conv2d(): bias must have shape (C_out) or (1, C_out).");
            }

            const bool use_bias = (bias != nullptr);
            const bool requires_grad =
                input.requires_grad() || weight.requires_grad() || (use_bias && bias->requires_grad());

            Tensor out = Tensor::zeros(Shape({n, c_out, h_out, w_out}), requires_grad);

            for (Size b = 0; b < n; ++b)
            {
                for (Size co = 0; co < c_out; ++co)
                {
                    const Size gidx = co / c_out_per_group;
                    const Size ci_start = gidx * c_in_per_group;
                    for (Size oh = 0; oh < h_out; ++oh)
                    {
                        for (Size ow = 0; ow < w_out; ++ow)
                        {
                            Tensor::value_type acc = 0.0f;
                            for (Size ci_local = 0; ci_local < c_in_per_group; ++ci_local)
                            {
                                const Size ci = ci_start + ci_local;
                                for (Size kh = 0; kh < k_h; ++kh)
                                {
                                    const long long ih = static_cast<long long>(oh * cfg.stride_h + kh * cfg.dilation_h) - static_cast<long long>(cfg.pad_h);
                                    if (ih < 0 || ih >= static_cast<long long>(h_in))
                                    {
                                        continue;
                                    }

                                    for (Size kw = 0; kw < k_w; ++kw)
                                    {
                                        const long long iw = static_cast<long long>(ow * cfg.stride_w + kw * cfg.dilation_w) - static_cast<long long>(cfg.pad_w);
                                        if (iw < 0 || iw >= static_cast<long long>(w_in))
                                        {
                                            continue;
                                        }

                                        acc += input.at({b, ci, static_cast<Size>(ih), static_cast<Size>(iw)}) *
                                               weight.at({co, ci_local, kh, kw});
                                    }
                                }
                            }

                            if (use_bias)
                            {
                                acc += bias_value(*bias, co);
                            }

                            out.at({b, co, oh, ow}) = acc;
                        }
                    }
                }
            }

            out.set_leaf(!requires_grad);
            out.set_requires_grad(requires_grad);
            if (requires_grad)
            {
                const Tensor empty_bias = Tensor::zeros(Shape({0}), false);
                out.set_grad_fn(std::make_shared<Conv2dNode>(
                    input,
                    weight,
                    use_bias ? *bias : empty_bias,
                    use_bias,
                    cfg));
            }

            return out;
        }

    } // namespace

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  Size stride_h,
                  Size stride_w,
                  Size pad_h,
                  Size pad_w,
                  Size dilation_h,
                  Size dilation_w,
                  Size groups)
    {
        return conv2d_impl(input, weight, nullptr, {stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups});
    }

    Tensor conv2d(const Tensor &input,
                  const Tensor &weight,
                  const Tensor &bias,
                  Size stride_h,
                  Size stride_w,
                  Size pad_h,
                  Size pad_w,
                  Size dilation_h,
                  Size dilation_w,
                  Size groups)
    {
        return conv2d_impl(input, weight, &bias, {stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups});
    }

} // namespace synara
