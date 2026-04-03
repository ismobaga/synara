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
        constexpr Tensor::value_type kBeta = 0.044715f;

        Tensor grad_a = Tensor::zeros(a_.shape(), false);

        for (Size i = 0; i < a_.numel(); ++i)
        {
            const Tensor::value_type x = a_.data()[i];
            const Tensor::value_type inner = kAlpha * (x + kBeta * x * x * x);
            const Tensor::value_type t = std::tanh(inner);
            const Tensor::value_type sech2 = 1.0f - t * t;
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

    // ---- Dim-aware Reduction Nodes ----
    SumDimNode::SumDimNode(const Tensor &a, int dim, bool keepdim)
        : a_(a), dim_(dim), keepdim_(keepdim) {}
    void SumDimNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        // grad_output has dim removed (or size-1 if keepdim)
        // We need to broadcast it back to a_'s shape by repeating along dim_
        const Shape &in_shape = a_.shape();
        Size dim_size = in_shape[dim_];
        // If not keepdim, we need to insert dimension first
        Tensor expanded = keepdim_ ? grad_output : grad_output.unsqueeze(dim_);
        // Broadcast: repeat along dim_ to match a_.shape()
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        const auto &out_dims = in_shape.dims();
        Size outer = 1;
        for (int i = 0; i < dim_; ++i)
            outer *= out_dims[i];
        Size inner = 1;
        for (Size i = static_cast<Size>(dim_) + 1; i < out_dims.size(); ++i)
            inner *= out_dims[i];
        const Tensor::value_type *gdata = expanded.data();
        Tensor::value_type *odata = grad_a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    odata[o * dim_size * inner + d * inner + i] =
                        gdata[o * inner + i];
                }
            }
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    MeanDimNode::MeanDimNode(const Tensor &a, int dim, bool keepdim)
        : a_(a), dim_(dim), keepdim_(keepdim) {}
    void MeanDimNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        const Shape &in_shape = a_.shape();
        Size dim_size = in_shape[dim_];
        Tensor expanded = keepdim_ ? grad_output : grad_output.unsqueeze(dim_);
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        const auto &out_dims = in_shape.dims();
        Size outer = 1;
        for (int i = 0; i < dim_; ++i)
            outer *= out_dims[i];
        Size inner = 1;
        for (Size i = static_cast<Size>(dim_) + 1; i < out_dims.size(); ++i)
            inner *= out_dims[i];
        const Tensor::value_type scale = 1.0f / static_cast<Tensor::value_type>(dim_size);
        const Tensor::value_type *gdata = expanded.data();
        Tensor::value_type *odata = grad_a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    odata[o * dim_size * inner + d * inner + i] =
                        gdata[o * inner + i] * scale;
                }
            }
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    MaxDimNode::MaxDimNode(const Tensor &a, int dim, bool keepdim, const Tensor &output)
        : a_(a), dim_(dim), keepdim_(keepdim), output_(output) {}
    void MaxDimNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        const Shape &in_shape = a_.shape();
        Size dim_size = in_shape[dim_];
        Tensor expanded_out = keepdim_ ? output_ : output_.unsqueeze(dim_);
        Tensor expanded_grad = keepdim_ ? grad_output : grad_output.unsqueeze(dim_);
        const auto &out_dims = in_shape.dims();
        Size outer = 1;
        for (int i = 0; i < dim_; ++i)
            outer *= out_dims[i];
        Size inner = 1;
        for (Size i = static_cast<Size>(dim_) + 1; i < out_dims.size(); ++i)
            inner *= out_dims[i];
        // Count ties per (outer, inner) position
        std::vector<Size> tie_counts(outer * inner, 0);
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    if (a_.data()[o * dim_size * inner + d * inner + i] ==
                        expanded_out.data()[o * inner + i])
                    {
                        tie_counts[o * inner + i]++;
                    }
                }
            }
        }
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    Size flat_in = o * dim_size * inner + d * inner + i;
                    Size flat_out = o * inner + i;
                    if (a_.data()[flat_in] == expanded_out.data()[flat_out])
                    {
                        grad_a.data()[flat_in] = expanded_grad.data()[flat_out] /
                                                 static_cast<Tensor::value_type>(tie_counts[flat_out]);
                    }
                }
            }
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    MinDimNode::MinDimNode(const Tensor &a, int dim, bool keepdim, const Tensor &output)
        : a_(a), dim_(dim), keepdim_(keepdim), output_(output) {}
    void MinDimNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        const Shape &in_shape = a_.shape();
        Size dim_size = in_shape[dim_];
        Tensor expanded_out = keepdim_ ? output_ : output_.unsqueeze(dim_);
        Tensor expanded_grad = keepdim_ ? grad_output : grad_output.unsqueeze(dim_);
        const auto &out_dims = in_shape.dims();
        Size outer = 1;
        for (int i = 0; i < dim_; ++i)
            outer *= out_dims[i];
        Size inner = 1;
        for (Size i = static_cast<Size>(dim_) + 1; i < out_dims.size(); ++i)
            inner *= out_dims[i];
        std::vector<Size> tie_counts(outer * inner, 0);
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    if (a_.data()[o * dim_size * inner + d * inner + i] ==
                        expanded_out.data()[o * inner + i])
                    {
                        tie_counts[o * inner + i]++;
                    }
                }
            }
        }
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size o = 0; o < outer; ++o)
        {
            for (Size d = 0; d < dim_size; ++d)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    Size flat_in = o * dim_size * inner + d * inner + i;
                    Size flat_out = o * inner + i;
                    if (a_.data()[flat_in] == expanded_out.data()[flat_out])
                    {
                        grad_a.data()[flat_in] = expanded_grad.data()[flat_out] /
                                                 static_cast<Tensor::value_type>(tie_counts[flat_out]);
                    }
                }
            }
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    // ---- Shape Nodes ----
    SqueezeNode::SqueezeNode(const Tensor &a, int dim)
        : a_(a), dim_(dim) {}
    void SqueezeNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        // Inverse of squeeze is unsqueeze
        Tensor grad_a = grad_output.reshape(a_.shape());
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    UnsqueezeNode::UnsqueezeNode(const Tensor &a, int dim)
        : a_(a), dim_(dim) {}
    void UnsqueezeNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        // Inverse of unsqueeze is reshape back to original shape
        Tensor grad_a = grad_output.reshape(a_.shape());
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    PermuteNode::PermuteNode(const Tensor &a, const std::vector<int> &dims)
        : a_(a), dims_(dims) {}
    void PermuteNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        // Compute inverse permutation
        int r = static_cast<int>(dims_.size());
        std::vector<int> inv_perm(r);
        for (int i = 0; i < r; ++i)
        {
            int nd = dims_[i];
            if (nd < 0)
                nd += r;
            inv_perm[nd] = i;
        }
        Tensor grad_a = grad_output.permute(inv_perm);
        // grad_a might not be contiguous, need contiguous copy
        Tensor grad_a_cont = Tensor::zeros(a_.shape(), false);
        // Copy element by element using indices
        Size n = a_.numel();
        const auto &shape = a_.shape();
        int rank = static_cast<int>(shape.rank());
        std::vector<Size> idx(rank, 0);
        for (Size flat = 0; flat < n; ++flat)
        {
            // Compute multi-index for flat in a_'s shape
            Size tmp = flat;
            for (int d = rank - 1; d >= 0; --d)
            {
                idx[d] = tmp % shape[d];
                tmp /= shape[d];
            }
            // grad_a has the permuted shape, access via its at()
            grad_a_cont.data()[flat] = grad_a.at(std::vector<Size>(idx.begin(), idx.end()));
        }
        a_.accumulate_grad(grad_a_cont);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a_cont);
    }
    CatNode::CatNode(const std::vector<Tensor> &inputs, int dim)
        : inputs_(inputs), dim_(dim) {}
    void CatNode::backward(const Tensor &grad_output)
    {
        // Split grad_output along dim_ into pieces matching each input
        Size offset = 0;
        const auto &out_shape = grad_output.shape();
        int r = static_cast<int>(out_shape.rank());
        int d = dim_;
        if (d < 0)
            d += r;
        for (auto &inp : inputs_)
        {
            if (!inp.requires_grad())
            {
                offset += inp.shape()[d];
                continue;
            }
            Size seg_size = inp.shape()[d];
            // Extract slice along dim d from grad_output
            Tensor grad_piece = Tensor::zeros(inp.shape(), false);
            // Compute outer/inner sizes
            Size outer = 1;
            for (int i = 0; i < d; ++i)
                outer *= out_shape[i];
            Size total_d = out_shape[d];
            Size inner = 1;
            for (int i = d + 1; i < r; ++i)
                inner *= out_shape[i];
            const Tensor::value_type *gdata = grad_output.data();
            Tensor::value_type *pdata = grad_piece.data();
            for (Size o = 0; o < outer; ++o)
            {
                for (Size k = 0; k < seg_size; ++k)
                {
                    for (Size i = 0; i < inner; ++i)
                    {
                        pdata[o * seg_size * inner + k * inner + i] =
                            gdata[o * total_d * inner + (offset + k) * inner + i];
                    }
                }
            }
            inp.accumulate_grad(grad_piece);
            if (inp.grad_fn())
                inp.grad_fn()->backward(grad_piece);
            offset += seg_size;
        }
    }
    StackNode::StackNode(const std::vector<Tensor> &inputs, int dim)
        : inputs_(inputs), dim_(dim) {}
    void StackNode::backward(const Tensor &grad_output)
    {
        // grad_output has an extra dim at position dim_
        // Select each slice along that dim, squeeze it, propagate
        int r = static_cast<int>(grad_output.shape().rank());
        int d = dim_;
        if (d < 0)
            d += r;
        const auto &out_shape = grad_output.shape();
        Size outer = 1;
        for (int i = 0; i < d; ++i)
            outer *= out_shape[i];
        Size inner = 1;
        for (int i = d + 1; i < r; ++i)
            inner *= out_shape[i];
        for (Size idx = 0; idx < inputs_.size(); ++idx)
        {
            auto &inp = inputs_[idx];
            if (!inp.requires_grad())
                continue;
            Tensor grad_piece = Tensor::zeros(inp.shape(), false);
            const Tensor::value_type *gdata = grad_output.data();
            Tensor::value_type *pdata = grad_piece.data();
            Size num_stacks = inputs_.size();
            for (Size o = 0; o < outer; ++o)
            {
                for (Size i = 0; i < inner; ++i)
                {
                    pdata[o * inner + i] =
                        gdata[o * num_stacks * inner + idx * inner + i];
                }
            }
            inp.accumulate_grad(grad_piece);
            if (inp.grad_fn())
                inp.grad_fn()->backward(grad_piece);
        }
    }
    SplitNode::SplitNode(const Tensor &a, int split_size, int dim)
        : a_(a), split_size_(split_size), dim_(dim) {}
    void SplitNode::backward(const Tensor &grad_output)
    {
        // This is called per-output; we need to accumulate into a_
        // grad_output corresponds to one piece; we don't know which piece here.
        // The SplitNode approach: each output has its own SplitPieceNode.
        // For simplicity, we don't use SplitNode::backward directly;
        // see the SplitPieceNode implementation below.
        // This backward is a no-op placeholder.
        (void)grad_output;
    }
    void SplitNode::register_output(std::size_t idx, Tensor &out)
    {
        (void)idx;
        (void)out;
    }
    // ---- Math Nodes ----
    ExpNode::ExpNode(const Tensor &a, const Tensor &output)
        : a_(a), output_(output) {}
    void ExpNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            grad_a.data()[i] = grad_output.data()[i] * output_.data()[i];
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    LogNode::LogNode(const Tensor &a) : a_(a) {}
    void LogNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            grad_a.data()[i] = grad_output.data()[i] / a_.data()[i];
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    Log2Node::Log2Node(const Tensor &a) : a_(a) {}
    void Log2Node::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        constexpr Tensor::value_type ln2 = 0.693147180559945f;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            grad_a.data()[i] = grad_output.data()[i] / (a_.data()[i] * ln2);
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    SqrtNode::SqrtNode(const Tensor &a, const Tensor &output)
        : a_(a), output_(output) {}
    void SqrtNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            grad_a.data()[i] = grad_output.data()[i] / (2.0f * output_.data()[i]);
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    PowNode::PowNode(const Tensor &a, Tensor::value_type exponent)
        : a_(a), exponent_(exponent) {}
    void PowNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            grad_a.data()[i] = grad_output.data()[i] * exponent_ *
                               std::pow(a_.data()[i], exponent_ - 1.0f);
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    AbsNode::AbsNode(const Tensor &a) : a_(a) {}
    void AbsNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            Tensor::value_type x = a_.data()[i];
            grad_a.data()[i] = grad_output.data()[i] * (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f));
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    SignNode::SignNode(const Tensor &a) : a_(a) {}
    void SignNode::backward(const Tensor & /*grad_output*/)
    {
        // sign is piecewise constant, gradient is 0 everywhere
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }
    ClampNode::ClampNode(const Tensor &a, Tensor::value_type min_val, Tensor::value_type max_val)
        : a_(a), min_val_(min_val), max_val_(max_val) {}
    void ClampNode::backward(const Tensor &grad_output)
    {
        if (!a_.requires_grad())
            return;
        Tensor grad_a = Tensor::zeros(a_.shape(), false);
        for (Size i = 0; i < a_.numel(); ++i)
        {
            Tensor::value_type x = a_.data()[i];
            grad_a.data()[i] = (x >= min_val_ && x <= max_val_) ? grad_output.data()[i] : 0.0f;
        }
        a_.accumulate_grad(grad_a);
        if (a_.grad_fn())
            a_.grad_fn()->backward(grad_a);
    }

    EmbeddingNode::EmbeddingNode(Tensor indices, Tensor weight)
        : indices_(std::move(indices)), weight_(std::move(weight)) {}

    void EmbeddingNode::backward(const Tensor &grad_output)
    {
        if (!weight_.requires_grad())
            return;
        const Size n = indices_.numel();
        const Size embed_dim = weight_.shape()[1];
        Tensor grad_w = Tensor::zeros(weight_.shape(), false);
        const Tensor::value_type *go = grad_output.data();
        Tensor::value_type *gw = grad_w.data();
        for (Size i = 0; i < n; ++i)
        {
            const auto idx = static_cast<Size>(indices_.data()[i]);
            for (Size d = 0; d < embed_dim; ++d)
                gw[idx * embed_dim + d] += go[i * embed_dim + d];
        }
        weight_.accumulate_grad(grad_w);
        if (weight_.grad_fn())
            weight_.grad_fn()->backward(grad_w);
    }
} // namespace synara