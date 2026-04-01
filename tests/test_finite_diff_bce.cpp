#include <cassert>
#include <cmath>

#include "synara/ops/loss.hpp"
#include <vector>
#include <iostream>

int main()
{
    using namespace synara;

    // Create predictions with requires_grad enabled
    auto predictions = Tensor::from_vector(Shape({4, 1}),
                                           {0.1f, 0.3f, 0.7f, 0.9f},
                                           true);

    // Create fixed targets
    auto targets = Tensor::from_vector(Shape({4, 1}),
                                       {0.0f, 0.0f, 1.0f, 1.0f});

    // Compute analytical gradients via backprop
    auto loss = binary_cross_entropy(predictions, targets);
    loss.backward();

    // Copy analytical gradients before they're modified
    std::vector<float> pred_grad_data(predictions.grad().data(),
                                      predictions.grad().data() + predictions.grad().numel());

    // Finite difference step size (slightly larger for numerical stability with BCE)
    const float eps = 1e-4f;

    // Tolerance for BCE - looser due to log/exp operations
    const float abs_tol = 1e-3f;
    const float rel_tol = 5e-2f; // 5% relative tolerance (BCE is numerically sensitive)

    // Validate prediction gradients via finite differences
    for (size_t i = 0; i < predictions.numel(); i++)
    {
        // Forward difference: f(p + eps)
        float original = predictions.data()[i];
        predictions.data()[i] = original + eps;
        float f_plus = binary_cross_entropy(predictions, targets).item();

        // Backward difference: f(p - eps)
        predictions.data()[i] = original - eps;
        float f_minus = binary_cross_entropy(predictions, targets).item();

        // Restore predictions
        predictions.data()[i] = original;

        // Finite difference approximation
        float fd_grad = (f_plus - f_minus) / (2.0f * eps);

        // Analytical gradient
        float analytical = pred_grad_data[i];

        // Check match: use both absolute and relative tolerance
        float abs_error = std::fabs(fd_grad - analytical);
        float rel_error = abs_error / (std::fabs(analytical) + 1e-8f);

        // Pass if either absolute tolerance is met OR relative tolerance is met
        assert(abs_error < abs_tol || rel_error < rel_tol);
    }

    return 0;
}
