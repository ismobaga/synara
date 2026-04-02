#include "synara/autograd/nodes.hpp"

#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"
#include "synara/ops/linalg.hpp"

#include <cmath>

namespace synara
{

    AddNode::AddNode(const Tensor &a, const Tensor &b)
        : a_(a), b_(b) {}

    void AddNode::backward(const Tensor &grad_output)
    {
        if (a_.requires_grad())
        {
            a_.accumulate_grad(grad_output);
            if (a_.grad_fn())
            {
                a_.grad_fn()->backward(grad_output);
            }
        }

        if (b_.requires_grad())
        {
            b_.accumulate_grad(grad_output);
            if (b_.grad_fn())
            {
                b_.grad_fn()->backward(grad_output);
            }
        }
    }

    SubNode::SubNode(const Tensor &a, const Tensor &b)
        : a_(a), b_(b) {}

    void SubNode::backward(const Tensor &grad_output)
    {
        if (a_.requires_grad())
        {
            a_.accumulate_grad(grad_output);
            if (a_.grad_fn())
            {
                a_.grad_fn()->backward(grad_output);
            }
        }
        if (b_.requires_grad())
        {
            Tensor neg_grad = mul(grad_output, -1.0f);
            b_.accumulate_grad(neg_grad);
            if (b_.grad_fn())
            {
                b_.grad_fn()->backward(neg_grad);
            }
        }
    }

    MulNode::MulNode(const Tensor &a, const Tensor &b)
        : a_(a), b_(b) {}

    void MulNode::backward(const Tensor &grad_output)
    {
        if (a_.requires_grad())
        {
            Tensor grad_a = mul(grad_output, b_);
            a_.accumulate_grad(grad_a);
            if (a_.grad_fn())
            {
                a_.grad_fn()->backward(grad_a);
            }
        }

        if (b_.requires_grad())
        {
            Tensor grad_b = mul(grad_output, a_);
            b_.accumulate_grad(grad_b);
            if (b_.grad_fn())
            {
                b_.grad_fn()->backward(grad_b);
            }
        }
    }

    DivNode::DivNode(const Tensor &a, const Tensor &b)
        : a_(a), b_(b) {}

    void DivNode::backward(const Tensor &grad_output)
    {
        if (a_.requires_grad())
        {
            Tensor grad_a = div(grad_output, b_);
            a_.accumulate_grad(grad_a);
            if (a_.grad_fn())
            {
                a_.grad_fn()->backward(grad_a);
            }
        }
        if (b_.requires_grad())
        {
            Tensor neg_grad = mul(grad_output, div(a_, mul(b_, b_)));
            b_.accumulate_grad(neg_grad);
            if (b_.grad_fn())
            {
                b_.grad_fn()->backward(neg_grad);
            }
        }
    }

    SumNode::SumNode(const Tensor &a)
        : a_(a) {}

    void SumNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Tensor grad_a = Tensor::full(a_.shape(), grad_output.item(), false);
        a_.accumulate_grad(grad_a);

        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    MeanNode::MeanNode(const Tensor &a)
        : a_(a) {}

    void MeanNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Tensor grad_a = Tensor::full(a_.shape(), grad_output.item() / a_.numel(), false);
        a_.accumulate_grad(grad_a);

        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    MatMulNode::MatMulNode(const Tensor &a, const Tensor &b)
        : a_(a), b_(b) {}

    void MatMulNode::backward(const Tensor &grad_output)
    {
        if (a_.requires_grad())
        {
            Tensor grad_a = matmul(grad_output, b_.transpose(0, 1));
            a_.accumulate_grad(grad_a);
            if (a_.grad_fn())
            {
                a_.grad_fn()->backward(grad_a);
            }
        }

        if (b_.requires_grad())
        {
            Tensor grad_b = matmul(a_.transpose(0, 1), grad_output);
            b_.accumulate_grad(grad_b);
            if (b_.grad_fn())
            {
                b_.grad_fn()->backward(grad_b);
            }
        }
    }

    ReLUNode::ReLUNode(const Tensor &a)
        : a_(a) {}

    void ReLUNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }
        Tensor grad_a = grad_output;
        for (Size i = 0; i < a_.numel(); ++i)
        {
            if (a_.data()[i] <= 0)
            {
                grad_a.data()[i] = 0.0f;
            }
        }
        a_.accumulate_grad(grad_a);

        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    LeakyReLUNode::LeakyReLUNode(const Tensor &a, Tensor::value_type negative_slope)
        : a_(a), negative_slope_(negative_slope) {}

    void LeakyReLUNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Tensor grad_a = grad_output;
        for (Size i = 0; i < a_.numel(); ++i)
        {
            if (a_.data()[i] <= 0.0f)
            {
                grad_a.data()[i] = grad_output.data()[i] * negative_slope_;
            }
        }

        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    SigmoidNode::SigmoidNode(const Tensor &a)
        : a_(a) {}

    void SigmoidNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            const Tensor::value_type s = 1.0f / (1.0f + std::exp(-a_.data()[i]));
            grad_a.data()[i] = grad_output.data()[i] * s * (1.0f - s);
        }

        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    TanhNode::TanhNode(const Tensor &a)
        : a_(a) {}

    void TanhNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            const Tensor::value_type t = std::tanh(a_.data()[i]);
            grad_a.data()[i] = grad_output.data()[i] * (1.0f - t * t);
        }

        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    SoftmaxNode::SoftmaxNode(const Tensor &a, int dim, const Tensor &output)
        : a_(a), dim_(dim), output_(output) {}

    void SoftmaxNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        Shape shape = a_.shape();
        int rank = static_cast<int>(shape.rank());
        Size stride = 1;
        for (int i = dim_ + 1; i < rank; ++i)
        {
            stride *= shape[i];
        }
        Size outer_size = a_.numel() / (shape[dim_] * stride);

        Tensor grad_a = Tensor::zeros(a_.shape(), false);

        for (Size o = 0; o < outer_size; ++o)
        {
            for (Size s = 0; s < stride; ++s)
            {
                for (Size i = 0; i < shape[dim_]; ++i)
                {
                    Size idx_i = o * shape[dim_] * stride + i * stride + s;
                    Tensor::value_type sum_grad = 0.0f;

                    for (Size j = 0; j < shape[dim_]; ++j)
                    {
                        Size idx_j = o * shape[dim_] * stride + j * stride + s;
                        Tensor::value_type factor = output_.data()[idx_j] * ((i == j ? 1.0f : 0.0f) - output_.data()[idx_i]);
                        sum_grad += grad_output.data()[idx_j] * factor;
                    }
                    grad_a.data()[idx_i] = sum_grad;
                }
            }
        }

        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    BCENode::BCENode(const Tensor &pred, const Tensor &target, Tensor::value_type eps)
        : pred_(pred), target_(target), eps_(eps) {}

    void BCENode::backward(const Tensor &grad_output)
    {
        if (!pred_.requires_grad())
        {
            return;
        }

        const Tensor::value_type upstream = grad_output.item();
        Tensor grad_pred = Tensor::zeros(pred_.shape(), false);
        const Tensor::value_type n = static_cast<Tensor::value_type>(pred_.numel());

        for (Size i = 0; i < pred_.numel(); ++i)
        {
            const Tensor::value_type p = std::fmax(eps_, std::fmin(1.0f - eps_, pred_.data()[i]));
            const Tensor::value_type y = target_.data()[i];
            grad_pred.data()[i] = upstream * ((p - y) / (p * (1.0f - p) * n));
        }

        pred_.accumulate_grad(grad_pred);
        if (pred_.grad_fn())
        {
            pred_.grad_fn()->backward(grad_pred);
        }
    }

    GELUNode::GELUNode(const Tensor &a)
        : a_(a) {}

    void GELUNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
        {
            return;
        }

        constexpr Tensor::value_type kAlpha = 0.7978845608f; // sqrt(2/pi)
        constexpr Tensor::value_type kBeta  = 0.044715f;

        Tensor grad_a = Tensor::zeros(a_.shape(), false);

        for (Size i = 0; i < a_.numel(); ++i)
        {
            const Tensor::value_type x      = a_.data()[i];
            const Tensor::value_type inner  = kAlpha * (x + kBeta * x * x * x);
            const Tensor::value_type t      = std::tanh(inner);
            const Tensor::value_type sech2  = 1.0f - t * t;
            // d/dx GELU(x) = 0.5*(1 + t) + 0.5*x*sech^2(inner)*alpha*(1 + 3*beta*x^2)
            const Tensor::value_type d_inner = kAlpha * (1.0f + 3.0f * kBeta * x * x);
            grad_a.data()[i] = grad_output.data()[i] *
                               (0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner);
        }

        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
        {
            a_.grad_fn()->backward(grad_a);
        }
    }

    CrossEntropyNode::CrossEntropyNode(const Tensor &logits, const Tensor &targets)
        : logits_(logits), targets_(targets) {}

    void CrossEntropyNode::backward(const Tensor &grad_output)
    {
        if (!logits_.requires_grad())
        {
            return;
        }

        const Tensor::value_type upstream = grad_output.item();
        const Size N = logits_.shape()[0];
        const Size C = logits_.shape()[1];

        Tensor grad_logits = Tensor::zeros(logits_.shape(), false);

        for (Size n = 0; n < N; ++n)
        {
            // Compute softmax for sample n
            Tensor::value_type max_val = logits_.data()[n * C];
            for (Size c = 0; c < C; ++c)
            {
                if (logits_.data()[n * C + c] > max_val)
                    max_val = logits_.data()[n * C + c];
            }

            Tensor::value_type sum_exp = 0.0f;
            for (Size c = 0; c < C; ++c)
                sum_exp += std::exp(logits_.data()[n * C + c] - max_val);

            for (Size c = 0; c < C; ++c)
            {
                const Tensor::value_type softmax_c =
                    std::exp(logits_.data()[n * C + c] - max_val) / sum_exp;
                // gradient: (softmax - target) / N
                grad_logits.data()[n * C + c] =
                    upstream * (softmax_c - targets_.data()[n * C + c]) /
                    static_cast<Tensor::value_type>(N);
            }
        }

        logits_.accumulate_grad(grad_logits);
        if (logits_.grad_fn())
        {
            logits_.grad_fn()->backward(grad_logits);
        }
    }
} // namespace synara