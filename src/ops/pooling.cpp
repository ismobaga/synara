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

        bool should_parallelize_pool(Size n,
                                     Size c,
                                     Size h_out,
                                     Size w_out,
                                     Size kernel_h,
                                     Size kernel_w)
        {
            return static_cast<long long>(n) *
                       static_cast<long long>(c) *
                       static_cast<long long>(h_out) *
                       static_cast<long long>(w_out) *
                       static_cast<long long>(kernel_h) *
                       static_cast<long long>(kernel_w) >=
                   (1LL << 15);
        }

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
                const Size n = input_.shape()[0];
                const Size c = input_.shape()[1];
                const Size h_out = output_shape_[2];
                const Size w_out = output_shape_[3];
                const Size plane_size = h_out * w_out;
                const bool parallel = should_parallelize_pool(n, c, h_out, w_out, 1, 1);
                Tensor::value_type *grad_input_data = grad_input.data();
                const Tensor::value_type *grad_output_data = grad_output.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
                {
                    const Size out_base = static_cast<Size>(bc) * plane_size;
                    for (Size i = 0; i < plane_size; ++i)
                    {
                        const Size in_idx = argmax_[out_base + i];
                        if (in_idx < grad_input.numel())
                        {
                            grad_input_data[in_idx] += grad_output_data[out_base + i];
                        }
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

        class AvgPool2dNode : public Node
        {
        public:
            AvgPool2dNode(Tensor input,
                          Shape output_shape,
                          std::vector<Size> counts,
                          Size kernel_h,
                          Size kernel_w,
                          Size stride_h,
                          Size stride_w,
                          Size pad_h,
                          Size pad_w)
                : input_(std::move(input)),
                  output_shape_(std::move(output_shape)),
                  counts_(std::move(counts)),
                  kernel_h_(kernel_h),
                  kernel_w_(kernel_w),
                  stride_h_(stride_h),
                  stride_w_(stride_w),
                  pad_h_(pad_h),
                  pad_w_(pad_w)
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
                    throw ShapeError("AvgPool2dNode::backward(): gradient shape mismatch.");
                }

                const Size n = input_.shape()[0];
                const Size c = input_.shape()[1];
                const Size h = input_.shape()[2];
                const Size w = input_.shape()[3];
                const Size h_out = grad_output.shape()[2];
                const Size w_out = grad_output.shape()[3];

                Tensor grad_input = Tensor::zeros(input_.shape(), false);
                Tensor::value_type *grad_input_data = grad_input.data();
                const Tensor::value_type *grad_output_data = grad_output.data();
                const Size plane_size = h_out * w_out;
                const bool parallel = should_parallelize_pool(n, c, h_out, w_out, kernel_h_, kernel_w_);

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
                {
                    const Size batch_index = static_cast<Size>(bc) / c;
                    const Size channel_index = static_cast<Size>(bc) % c;
                    const Size out_base = static_cast<Size>(bc) * plane_size;
                    const Size channel_base = ((batch_index * c + channel_index) * h) * w;

                    for (Size oh = 0; oh < h_out; ++oh)
                    {
                        for (Size ow = 0; ow < w_out; ++ow)
                        {
                            const Size out_idx = out_base + oh * w_out + ow;
                            const Size count = counts_[out_idx];
                            const Tensor::value_type g = grad_output_data[out_idx] /
                                                         static_cast<Tensor::value_type>(count);

                            for (Size kh = 0; kh < kernel_h_; ++kh)
                            {
                                const long long ih = static_cast<long long>(oh * stride_h_ + kh) - static_cast<long long>(pad_h_);
                                if (ih < 0 || ih >= static_cast<long long>(h))
                                {
                                    continue;
                                }

                                const Size row_base = channel_base + static_cast<Size>(ih) * w;
                                for (Size kw = 0; kw < kernel_w_; ++kw)
                                {
                                    const long long iw = static_cast<long long>(ow * stride_w_ + kw) - static_cast<long long>(pad_w_);
                                    if (iw < 0 || iw >= static_cast<long long>(w))
                                    {
                                        continue;
                                    }

                                    grad_input_data[row_base + static_cast<Size>(iw)] += g;
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

        private:
            Tensor input_;
            Shape output_shape_;
            std::vector<Size> counts_;
            Size kernel_h_;
            Size kernel_w_;
            Size stride_h_;
            Size stride_w_;
            Size pad_h_;
            Size pad_w_;
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
        const Size plane_size = h_out * w_out;
        const bool parallel = should_parallelize_pool(n, c, h_out, w_out, kernel_h, kernel_w);

        if (input.is_contiguous())
        {
            const Tensor::value_type *input_data = input.data();
            Tensor::value_type *out_data = out.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
            {
                const Size batch_index = static_cast<Size>(bc) / c;
                const Size channel_index = static_cast<Size>(bc) % c;
                const Size channel_base = ((batch_index * c + channel_index) * h) * w;
                const Size out_base = static_cast<Size>(bc) * plane_size;

                for (Size oh = 0; oh < h_out; ++oh)
                {
                    const long long ih_origin = static_cast<long long>(oh * stride_h) - static_cast<long long>(pad_h);
                    for (Size ow = 0; ow < w_out; ++ow)
                    {
                        const Size out_idx = out_base + oh * w_out + ow;
                        const long long iw_origin = static_cast<long long>(ow * stride_w) - static_cast<long long>(pad_w);
                        Tensor::value_type best = -std::numeric_limits<Tensor::value_type>::infinity();
                        Size best_idx = input.numel();

                        for (Size kh = 0; kh < kernel_h; ++kh)
                        {
                            const long long ih = ih_origin + static_cast<long long>(kh);
                            if (ih < 0 || ih >= static_cast<long long>(h))
                            {
                                continue;
                            }

                            const Size row_base = channel_base + static_cast<Size>(ih) * w;
                            for (Size kw = 0; kw < kernel_w; ++kw)
                            {
                                const long long iw = iw_origin + static_cast<long long>(kw);
                                if (iw < 0 || iw >= static_cast<long long>(w))
                                {
                                    continue;
                                }

                                const Size idx = row_base + static_cast<Size>(iw);
                                const Tensor::value_type v = input_data[idx];
                                if (v > best)
                                {
                                    best = v;
                                    best_idx = idx;
                                }
                            }
                        }

                        out_data[out_idx] = best;
                        argmax[out_idx] = best_idx;
                    }
                }
            }
        }
        else
        {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
            {
                const Size batch_index = static_cast<Size>(bc) / c;
                const Size channel_index = static_cast<Size>(bc) % c;
                const Size out_base = static_cast<Size>(bc) * plane_size;

                for (Size oh = 0; oh < h_out; ++oh)
                {
                    for (Size ow = 0; ow < w_out; ++ow)
                    {
                        const Size out_idx = out_base + oh * w_out + ow;
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
                                const Tensor::value_type v = input.at({batch_index, channel_index, ihs, iws});
                                if (v > best)
                                {
                                    best = v;
                                    best_idx = ((batch_index * c + channel_index) * h + ihs) * w + iws;
                                }
                            }
                        }

                        out.at({batch_index, channel_index, oh, ow}) = best;
                        argmax[out_idx] = best_idx;
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

    Tensor avg_pool2d(const Tensor &input,
                      Size kernel_h,
                      Size kernel_w,
                      Size stride_h,
                      Size stride_w,
                      Size pad_h,
                      Size pad_w)
    {
        if (input.rank() != 4)
        {
            throw ShapeError("avg_pool2d(): input must be rank 4 (N, C, H, W).");
        }
        if (kernel_h == 0 || kernel_w == 0)
        {
            throw ValueError("avg_pool2d(): kernel dimensions must be > 0.");
        }
        if (stride_h == 0 || stride_w == 0)
        {
            throw ValueError("avg_pool2d(): stride must be >= 1.");
        }

        const Size n = input.shape()[0];
        const Size c = input.shape()[1];
        const Size h = input.shape()[2];
        const Size w = input.shape()[3];

        if (n == 0 || c == 0 || h == 0 || w == 0)
        {
            throw ShapeError("avg_pool2d(): all input dimensions must be > 0.");
        }

        const Size padded_h = h + 2 * pad_h;
        const Size padded_w = w + 2 * pad_w;
        if (padded_h < kernel_h || padded_w < kernel_w)
        {
            throw ShapeError("avg_pool2d(): kernel larger than padded input.");
        }

        const Size h_out = (padded_h - kernel_h) / stride_h + 1;
        const Size w_out = (padded_w - kernel_w) / stride_w + 1;

        const bool requires_grad = input.requires_grad();
        Tensor out = Tensor::zeros(Shape({n, c, h_out, w_out}), requires_grad);
        std::vector<Size> counts(out.numel(), 0);
        const Size plane_size = h_out * w_out;
        const bool parallel = should_parallelize_pool(n, c, h_out, w_out, kernel_h, kernel_w);

        if (input.is_contiguous())
        {
            const Tensor::value_type *input_data = input.data();
            Tensor::value_type *out_data = out.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
            {
                const Size batch_index = static_cast<Size>(bc) / c;
                const Size channel_index = static_cast<Size>(bc) % c;
                const Size channel_base = ((batch_index * c + channel_index) * h) * w;
                const Size out_base = static_cast<Size>(bc) * plane_size;

                for (Size oh = 0; oh < h_out; ++oh)
                {
                    const long long ih_origin = static_cast<long long>(oh * stride_h) - static_cast<long long>(pad_h);
                    for (Size ow = 0; ow < w_out; ++ow)
                    {
                        const Size out_idx = out_base + oh * w_out + ow;
                        const long long iw_origin = static_cast<long long>(ow * stride_w) - static_cast<long long>(pad_w);
                        Tensor::value_type acc = 0.0f;
                        Size count = 0;

                        for (Size kh = 0; kh < kernel_h; ++kh)
                        {
                            const long long ih = ih_origin + static_cast<long long>(kh);
                            if (ih < 0 || ih >= static_cast<long long>(h))
                            {
                                continue;
                            }

                            const Size row_base = channel_base + static_cast<Size>(ih) * w;
                            Size kw = 0;
                            for (; kw + 3 < kernel_w; kw += 4)
                            {
                                const long long iw0 = iw_origin + static_cast<long long>(kw);
                                const long long iw1 = iw0 + 1;
                                const long long iw2 = iw0 + 2;
                                const long long iw3 = iw0 + 3;

                                if (iw0 >= 0 && iw3 < static_cast<long long>(w))
                                {
                                    acc += input_data[row_base + static_cast<Size>(iw0)];
                                    acc += input_data[row_base + static_cast<Size>(iw1)];
                                    acc += input_data[row_base + static_cast<Size>(iw2)];
                                    acc += input_data[row_base + static_cast<Size>(iw3)];
                                    count += 4;
                                }
                                else
                                {
                                    if (iw0 >= 0 && iw0 < static_cast<long long>(w))
                                    {
                                        acc += input_data[row_base + static_cast<Size>(iw0)];
                                        ++count;
                                    }
                                    if (iw1 >= 0 && iw1 < static_cast<long long>(w))
                                    {
                                        acc += input_data[row_base + static_cast<Size>(iw1)];
                                        ++count;
                                    }
                                    if (iw2 >= 0 && iw2 < static_cast<long long>(w))
                                    {
                                        acc += input_data[row_base + static_cast<Size>(iw2)];
                                        ++count;
                                    }
                                    if (iw3 >= 0 && iw3 < static_cast<long long>(w))
                                    {
                                        acc += input_data[row_base + static_cast<Size>(iw3)];
                                        ++count;
                                    }
                                }
                            }
                            for (; kw < kernel_w; ++kw)
                            {
                                const long long iw = iw_origin + static_cast<long long>(kw);
                                if (iw < 0 || iw >= static_cast<long long>(w))
                                {
                                    continue;
                                }

                                acc += input_data[row_base + static_cast<Size>(iw)];
                                ++count;
                            }
                        }

                        counts[out_idx] = count;
                        out_data[out_idx] = acc / static_cast<Tensor::value_type>(count);
                    }
                }
            }
        }
        else
        {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long bc = 0; bc < static_cast<long long>(n * c); ++bc)
            {
                const Size batch_index = static_cast<Size>(bc) / c;
                const Size channel_index = static_cast<Size>(bc) % c;
                const Size out_base = static_cast<Size>(bc) * plane_size;

                for (Size oh = 0; oh < h_out; ++oh)
                {
                    for (Size ow = 0; ow < w_out; ++ow)
                    {
                        const Size out_idx = out_base + oh * w_out + ow;
                        Tensor::value_type acc = 0.0f;
                        Size count = 0;

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

                                acc += input.at({batch_index, channel_index, static_cast<Size>(ih), static_cast<Size>(iw)});
                                ++count;
                            }
                        }

                        counts[out_idx] = count;
                        out.data()[out_idx] = acc / static_cast<Tensor::value_type>(count);
                    }
                }
            }
        }

        out.set_leaf(!requires_grad);
        out.set_requires_grad(requires_grad);
        if (requires_grad)
        {
            out.set_grad_fn(std::make_shared<AvgPool2dNode>(
                input,
                out.shape(),
                std::move(counts),
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w));
        }

        return out;
    }

} // namespace synara
