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
} // namespace synara