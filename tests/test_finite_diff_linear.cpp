#include <cassert>
#include <cmath>

#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"
#include "synara/ops/reduction.hpp"
#include <iostream>
int main()
{
    using namespace synara;

    // Small network to validate Linear layer gradients via finite differences
    Linear layer(3, 2, true); // 3 inputs, 2 outputs, with bias

    // Create input and target
    auto x = Tensor::from_vector(Shape({2, 3}), {1.0f, 2.0f, 3.0f,
                                                 4.0f, 5.0f, 6.0f});
    auto target = Tensor::from_vector(Shape({2, 2}), {1.0f, 0.5f,
                                                      2.0f, 1.5f});

    // Compute analytical gradients via backprop
    auto pred = layer(x);
    auto loss = mse_loss(pred, target);
    loss.backward();

    auto params = layer.parameters();
    auto &weight = params[0]->tensor();
    auto &bias = params[1]->tensor();

    // Copy analytical gradients before they're modified
    std::vector<float> weight_grad_data(weight.grad().data(),
                                        weight.grad().data() + weight.grad().numel());
    std::vector<float> bias_grad_data(bias.grad().data(),
                                      bias.grad().data() + bias.grad().numel());

    // Finite difference step size
    const float eps = 1e-4f;

    // Tolerance: FD and analytical may differ due to numerical precision
    // Use both absolute and relative tolerance
    const float abs_tol = 1e-4f;
    const float rel_tol = 1e-2f; // 1% relative tolerance

    // Validate weight gradients via finite differences
    for (size_t i = 0; i < weight.numel(); i++)
    {
        // Forward difference: f(w + eps)
        float original = weight.data()[i];
        weight.data()[i] = original + eps;
        auto pred_plus = layer(x);
        auto loss_plus = mse_loss(pred_plus, target);
        float f_plus = loss_plus.item();

        // Backward difference: f(w - eps)
        weight.data()[i] = original - eps;
        auto pred_minus = layer(x);
        auto loss_minus = mse_loss(pred_minus, target);
        float f_minus = loss_minus.item();

        // Restore weight
        weight.data()[i] = original;

        // Finite difference approximation
        float fd_grad = (f_plus - f_minus) / (2.0f * eps);

        // Analytical gradient
        float analytical = weight_grad_data[i];

        // Check match: use both absolute and relative tolerance
        float rel_error = std::fabs(fd_grad - analytical) / (std::fabs(analytical) + 1e-8f);
        float abs_error = std::fabs(fd_grad - analytical);

        // Pass if either absolute tolerance is met OR relative tolerance is met
        assert(abs_error < abs_tol || rel_error < rel_tol);
    }

    // Validate bias gradients via finite differences
    for (size_t i = 0; i < bias.numel(); i++)
    {
        // Forward difference: f(b + eps)
        float original = bias.data()[i];
        bias.data()[i] = original + eps;
        auto pred_plus = layer(x);
        auto loss_plus = mse_loss(pred_plus, target);
        float f_plus = loss_plus.item();

        // Backward difference: f(b - eps)
        bias.data()[i] = original - eps;
        auto pred_minus = layer(x);
        auto loss_minus = mse_loss(pred_minus, target);
        float f_minus = loss_minus.item();

        // Restore bias
        bias.data()[i] = original;

        // Finite difference approximation
        float fd_grad = (f_plus - f_minus) / (2.0f * eps);

        // Analytical gradient
        float analytical = bias_grad_data[i];

        // Check match: use both absolute and relative tolerance
        float rel_error = std::fabs(fd_grad - analytical) / (std::fabs(analytical) + 1e-8f);
        float abs_error = std::fabs(fd_grad - analytical);

        // Pass if either absolute tolerance is met OR relative tolerance is met
        assert(abs_error < abs_tol || rel_error < rel_tol);
    }

    return 0;
}
