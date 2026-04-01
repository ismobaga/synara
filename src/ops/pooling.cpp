#include "synara/ops/pooling.hpp"

#include <limits>
#include <memory>
#include <vector>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        struct MaxPool2dConfig
        {
            Size kernel_h;
            Size kernel_w;
            Size stride_h;
            Size stride_w;
            Size pad_h;
            Size pad_w;
        };

        class MaxPool2dNode : public Node
        {
        public:
            MaxPool2dNode(Tensor input,
                          Shape output_shape,
                          std::vector<Size> argmax)
                : input_(std::move(input)),
                  output_shape_(std::move(output_shape)),
                  argmax_(std::move(argmax))
            {
            }

            void backward(const Tensor &grad_output) override
            {
                if (!input_.requires_grad())
                {
                    return;
                }

                if (grad_output.shape() != output_shape_)
                {
                    throw ShapeError("MaxPool2dNode::backward(): gradient shape mismatch.");
                }

                Tensor grad_input = Tensor::zeros(input_.shape(), false);
                for (Size i = 0; i < argmax_.size(); ++i)
                {
                    const Size in_idx = argmax_[i];
                    if (in_idx < grad_input.numel())
                    {
                        grad_input.data()[in_idx] += grad_output.data()[i];
                    }
                }

                input_.accumulate_grad(grad_input);
                if (input_.grad_fn())
                {
                    input_.grad_fn()->backward(grad_input);
                }
            }

        private:
            Tensor input_;
            Shape output_shape_;
            std::vector<Size> argmax_;
        };

    } // namespace

    Tensor max_pool2d(const Tensor &input,
                      Size kernel_h,
                      Size kernel_w,
                      Size stride_h,
                      Size stride_w,
                      Size pad_h,
                      Size pad_w)
    {
        if (input.rank() != 4)
        {
            throw ShapeError("max_pool2d(): input must be rank 4 (N, C, H, W).");
        }
        if (kernel_h == 0 || kernel_w == 0)
        {
            throw ValueError("max_pool2d(): kernel dimensions must be > 0.");
        }
        if (stride_h == 0 || stride_w == 0)
        {
            throw ValueError("max_pool2d(): stride must be >= 1.");
        }

        const Size n = input.shape()[0];
        const Size c = input.shape()[1];
        const Size h = input.shape()[2];
        const Size w = input.shape()[3];

        if (n == 0 || c == 0 || h == 0 || w == 0)
        {
            throw ShapeError("max_pool2d(): all input dimensions must be > 0.");
        }

        const Size padded_h = h + 2 * pad_h;
        const Size padded_w = w + 2 * pad_w;

        if (padded_h < kernel_h || padded_w < kernel_w)
        {
            throw ShapeError("max_pool2d(): kernel larger than padded input.");
        }

        const Size h_out = (padded_h - kernel_h) / stride_h + 1;
        const Size w_out = (padded_w - kernel_w) / stride_w + 1;

        const bool requires_grad = input.requires_grad();
        Tensor out = Tensor::zeros(Shape({n, c, h_out, w_out}), requires_grad);
        std::vector<Size> argmax(out.numel(), input.numel());

        Size out_flat = 0;
        for (Size b = 0; b < n; ++b)
        {
            for (Size ch = 0; ch < c; ++ch)
            {
                for (Size oh = 0; oh < h_out; ++oh)
                {
                    for (Size ow = 0; ow < w_out; ++ow)
                    {
                        Tensor::value_type best = -std::numeric_limits<Tensor::value_type>::infinity();
                        Size best_idx = input.numel();

                        for (Size kh = 0; kh < kernel_h; ++kh)
                        {
                            const long long ih = static_cast<long long>(oh * stride_h + kh) - static_cast<long long>(pad_h);
                            if (ih < 0 || ih >= static_cast<long long>(h))
                            {
                                continue;
                            }

                            for (Size kw = 0; kw < kernel_w; ++kw)
                            {
                                const long long iw = static_cast<long long>(ow * stride_w + kw) - static_cast<long long>(pad_w);
                                if (iw < 0 || iw >= static_cast<long long>(w))
                                {
                                    continue;
                                }

                                const Size ihs = static_cast<Size>(ih);
                                const Size iws = static_cast<Size>(iw);
                                const Tensor::value_type v = input.at({b, ch, ihs, iws});
                                if (v > best)
                                {
                                    best = v;
                                    best_idx = ((b * c + ch) * h + ihs) * w + iws;
                                }
                            }
                        }

                        out.at({b, ch, oh, ow}) = best;
                        argmax[out_flat++] = best_idx;
                    }
                }
            }
        }

        out.set_leaf(!requires_grad);
        out.set_requires_grad(requires_grad);
        if (requires_grad)
        {
            out.set_grad_fn(std::make_shared<MaxPool2dNode>(input, out.shape(), std::move(argmax)));
        }

        return out;
    }

} // namespace synara
