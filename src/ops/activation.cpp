#include "synara/ops/activation.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

#include <cmath>

namespace synara
{

    Tensor relu(const Tensor &a)
    {
        Tensor out = Tensor::zeros(a.shape());

        for (Size i = 0; i < a.numel(); ++i)
        {
            out.data()[i] = (a.data()[i] > 0.0f) ? a.data()[i] : 0.0f;
        }

        out.set_leaf(!a.requires_grad());
        out.set_requires_grad(a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<ReLUNode>(a);
            out.set_grad_fn(node);
        }

        return out;
    }

    Tensor leaky_relu(const Tensor &a, Tensor::value_type negative_slope)
    {
        Tensor out = Tensor::zeros(a.shape());

        for (Size i = 0; i < a.numel(); ++i)
        {
            out.data()[i] = (a.data()[i] > 0.0f) ? a.data()[i] : negative_slope * a.data()[i];
        }

        out.set_leaf(!a.requires_grad());
        out.set_requires_grad(a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<LeakyReLUNode>(a, negative_slope);
            out.set_grad_fn(node);
        }

        return out;
    }

    Tensor sigmoid(const Tensor &a)
    {
        Tensor out = Tensor::zeros(a.shape());

        for (Size i = 0; i < a.numel(); ++i)
        {
            out.data()[i] = 1.0f / (1.0f + std::exp(-a.data()[i]));
        }

        out.set_leaf(!a.requires_grad());
        out.set_requires_grad(a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<SigmoidNode>(a);
            out.set_grad_fn(node);
        }

        return out;
    }

    Tensor tanh(const Tensor &a)
    {
        Tensor out = Tensor::zeros(a.shape());

        for (Size i = 0; i < a.numel(); ++i)
        {
            out.data()[i] = std::tanh(a.data()[i]);
        }

        out.set_leaf(!a.requires_grad());
        out.set_requires_grad(a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<TanhNode>(a);
            out.set_grad_fn(node);
        }

        return out;
    }

    Tensor softmax(const Tensor &a, int dim)
    {
        Shape shape = a.shape();
        int rank = static_cast<int>(shape.rank());
        if (dim < 0)
            dim = rank + dim;
        if (dim < 0 || dim >= rank)
            throw std::out_of_range("softmax dim out of range");

        Tensor out = Tensor::zeros(shape, false);
        Size stride = 1;
        for (int i = dim + 1; i < rank; ++i)
        {
            stride *= shape[i];
        }
        Size outer_size = a.numel() / (shape[dim] * stride);

        for (Size o = 0; o < outer_size; ++o)
        {
            for (Size s = 0; s < stride; ++s)
            {
                Tensor::value_type max_val = a.data()[o * shape[dim] * stride + 0 * stride + s];
                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    max_val = std::max(max_val, a.data()[idx]);
                }

                Tensor::value_type sum_exp = 0.0f;
                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    Tensor::value_type exp_val = std::exp(a.data()[idx] - max_val);
                    out.data()[idx] = exp_val;
                    sum_exp += exp_val;
                }

                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    out.data()[idx] /= sum_exp;
                }
            }
        }

        out.set_leaf(!a.requires_grad());
        out.set_requires_grad(a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<SoftmaxNode>(a, dim, out);
            out.set_grad_fn(node);
        }

        return out;
    }

} // namespace synara