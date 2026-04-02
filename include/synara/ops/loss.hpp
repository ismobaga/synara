#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    Tensor mse_loss(const Tensor &pred, const Tensor &target);
    Tensor binary_cross_entropy(const Tensor &pred, const Tensor &target);

    /// Numerically-stable log-softmax along @p dim.
    /// Equivalent to log(softmax(a, dim)) but avoids overflow.
    Tensor log_softmax(const Tensor &a, int dim = -1);

    /// Cross-entropy loss for multi-class classification.
    ///
    /// @p logits  Raw (unnormalised) scores, shape (N, C).
    /// @p targets One-hot encoded target probabilities, shape (N, C).
    ///
    /// Computes  -sum(targets * log_softmax(logits)) / N  and supports
    /// autograd through @p logits.
    Tensor cross_entropy_loss(const Tensor &logits, const Tensor &targets);

} // namespace synara