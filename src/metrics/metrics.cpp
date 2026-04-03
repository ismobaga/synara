#include "synara/metrics/metrics.hpp"

#include "synara/core/error.hpp"

namespace synara
{

    double accuracy(const Tensor &predictions, const Tensor &targets)
    {
        if (predictions.rank() != 2)
        {
            throw ShapeError("accuracy(): predictions must be rank-2 [N, C].");
        }
        if (targets.rank() != 1)
        {
            throw ShapeError("accuracy(): targets must be rank-1 [N].");
        }

        const Size n = predictions.shape()[0];
        const Size c = predictions.shape()[1];
        if (targets.shape()[0] != n)
        {
            throw ShapeError("accuracy(): batch size mismatch.");
        }

        Size correct = 0;
        for (Size i = 0; i < n; ++i)
        {
            Size best_idx = 0;
            Tensor::value_type best = predictions.at({i, 0});
            for (Size j = 1; j < c; ++j)
            {
                const Tensor::value_type v = predictions.at({i, j});
                if (v > best)
                {
                    best = v;
                    best_idx = j;
                }
            }

            const Size target_idx = static_cast<Size>(targets.data()[i]);
            if (best_idx == target_idx)
            {
                ++correct;
            }
        }

        return (n == 0) ? 0.0 : static_cast<double>(correct) / static_cast<double>(n);
    }

    double binary_accuracy(const Tensor &predictions, const Tensor &targets, Tensor::value_type threshold)
    {
        if (predictions.numel() != targets.numel())
        {
            throw ShapeError("binary_accuracy(): predictions and targets must have same numel.");
        }

        Size correct = 0;
        for (Size i = 0; i < predictions.numel(); ++i)
        {
            const Tensor::value_type pred = predictions.data()[i] >= threshold ? 1.0f : 0.0f;
            const Tensor::value_type target = targets.data()[i] >= 0.5f ? 1.0f : 0.0f;
            if (pred == target)
            {
                ++correct;
            }
        }

        return predictions.numel() == 0
                   ? 0.0
                   : static_cast<double>(correct) / static_cast<double>(predictions.numel());
    }

} // namespace synara
