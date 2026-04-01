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

} // namespace synara