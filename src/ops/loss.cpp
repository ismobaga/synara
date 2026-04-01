#include "synara/ops/loss.hpp"
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"

namespace synara
{

    Tensor mse_loss(const Tensor &pred, const Tensor &target)
    {
        Tensor diff = sub(pred, target);
        return mean(mul(diff, diff));
    }

} // namespace synara