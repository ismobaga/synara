#include "synara/ops/loss.hpp"

#include <cmath>

#include "synara/autograd/nodes.hpp"
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"
#include "synara/core/error.hpp"

namespace synara
{

    Tensor mse_loss(const Tensor &pred, const Tensor &target)
    {
        Tensor diff = sub(pred, target);
        return mean(mul(diff, diff));
    }

    Tensor binary_cross_entropy(const Tensor &pred, const Tensor &target)
    {
        if (pred.shape() != target.shape())
        {
            throw ShapeError("binary_cross_entropy(): pred and target must have the same shape.");
        }

        constexpr Tensor::value_type eps = 1e-7f;
        Tensor::value_type total = 0.0f;
        for (Size i = 0; i < pred.numel(); ++i)
        {
            const Tensor::value_type p = std::fmax(eps, std::fmin(1.0f - eps, pred.data()[i]));
            const Tensor::value_type y = target.data()[i];
            total += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }

        Tensor out = Tensor::from_vector(Shape({}), {total / static_cast<Tensor::value_type>(pred.numel())}, pred.requires_grad());
        out.set_leaf(!pred.requires_grad());
        if (pred.requires_grad())
        {
            out.set_grad_fn(std::make_shared<BCENode>(pred, target, eps));
        }
        return out;
    }

    Tensor log_softmax(const Tensor &a, int dim)
    {
        Shape shape = a.shape();
        int rank = static_cast<int>(shape.rank());
        if (dim < 0)
            dim = rank + dim;
        if (dim < 0 || dim >= rank)
            throw std::out_of_range("log_softmax: dim out of range");

        Tensor out = Tensor::zeros(shape, false);

        Size stride = 1;
        for (int i = dim + 1; i < rank; ++i)
            stride *= shape[i];
        Size outer_size = a.numel() / (shape[dim] * stride);

        for (Size o = 0; o < outer_size; ++o)
        {
            for (Size s = 0; s < stride; ++s)
            {
                // Numerically stable: subtract max before exp
                Tensor::value_type max_val = a.data()[o * shape[dim] * stride + s];
                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    if (a.data()[idx] > max_val)
                        max_val = a.data()[idx];
                }

                Tensor::value_type log_sum_exp = 0.0f;
                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    log_sum_exp += std::exp(a.data()[idx] - max_val);
                }
                log_sum_exp = max_val + std::log(log_sum_exp);

                for (Size i = 0; i < shape[dim]; ++i)
                {
                    Size idx = o * shape[dim] * stride + i * stride + s;
                    out.data()[idx] = a.data()[idx] - log_sum_exp;
                }
            }
        }

        // log_softmax is differentiable but gradient flows through cross_entropy
        // when used via cross_entropy_loss; return without grad_fn here.
        out.set_leaf(true);
        out.set_requires_grad(false);
        return out;
    }

    Tensor cross_entropy_loss(const Tensor &logits, const Tensor &targets)
    {
        if (logits.shape() != targets.shape())
        {
            throw ShapeError("cross_entropy_loss(): logits and targets must have the same shape.");
        }
        if (logits.shape().rank() != 2)
        {
            throw ShapeError("cross_entropy_loss(): expected 2-D inputs (N, C).");
        }

        const Size N = logits.shape()[0];
        const Size C = logits.shape()[1];

        Tensor::value_type total = 0.0f;
        for (Size n = 0; n < N; ++n)
        {
            // stable log-softmax for sample n
            Tensor::value_type max_val = logits.data()[n * C];
            for (Size c = 0; c < C; ++c)
            {
                if (logits.data()[n * C + c] > max_val)
                    max_val = logits.data()[n * C + c];
            }

            Tensor::value_type sum_exp = 0.0f;
            for (Size c = 0; c < C; ++c)
                sum_exp += std::exp(logits.data()[n * C + c] - max_val);
            const Tensor::value_type log_sum_exp = max_val + std::log(sum_exp);

            for (Size c = 0; c < C; ++c)
            {
                const Tensor::value_type log_p = logits.data()[n * C + c] - log_sum_exp;
                total -= targets.data()[n * C + c] * log_p;
            }
        }

        Tensor out = Tensor::from_vector(Shape({}),
                                         {total / static_cast<Tensor::value_type>(N)},
                                         logits.requires_grad());
        out.set_leaf(!logits.requires_grad());
        if (logits.requires_grad())
        {
            out.set_grad_fn(std::make_shared<CrossEntropyNode>(logits, targets));
        }
        return out;
    }

} // namespace synara