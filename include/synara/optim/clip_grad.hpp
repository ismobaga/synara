#pragma once

#include <vector>

#include "synara/tensor/tensor.hpp"

namespace synara
{

    /// Clip the total gradient norm of a list of parameters in-place.
    ///
    /// The global L2 norm is computed across all gradients:
    ///   total_norm = sqrt(sum_p(sum_i(g_i^2)))
    ///
    /// If total_norm exceeds @p max_norm, every gradient is scaled by
    ///   max_norm / total_norm
    /// so that the combined norm equals exactly @p max_norm.
    ///
    /// @param params    Tensors whose gradients should be clipped.
    /// @param max_norm  Maximum allowed gradient norm (must be > 0).
    /// @returns         The total (pre-clipping) gradient norm.
    float clip_grad_norm(const std::vector<Tensor *> &params, float max_norm);

} // namespace synara
