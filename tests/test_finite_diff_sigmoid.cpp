#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/loss.hpp"
#include "synara/ops/reduction.hpp"
#include <iostream>

int main()
{
    using namespace synara;

    // Create input with requires_grad enabled
    auto input = Tensor::from_vector(Shape({3, 2}),
                                     {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                                     true);

    // Compute analytical gradients via backprop
    auto sig_output = sigmoid(input);
    auto sum_output = sum(sig_output); // Scalar output for backward()
    sum_output.backward();

    // Copy analytical gradients before they're modified
    std::vector<float> input_grad_data(input.grad().data(),
                                       input.grad().data() + input.grad().numel());

    // Finite difference step size
    const float eps = 1e-4f;

    // Tolerance for sigmoid - uses looser bounds due to nonlinearity
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-1f; // 10% relative tolerance for sigmoid

    // Validate input gradients via finite differences
    for (size_t i = 0; i < input.numel(); i++)
    {
        // Forward difference: f(x + eps)
        float original = input.data()[i];
        input.data()[i] = original + eps;
        auto sig_plus = sigmoid(input);
        float f_plus = sum(sig_plus).item();

        // Backward difference: f(x - eps)
        input.data()[i] = original - eps;
        auto sig_minus = sigmoid(input);
        float f_minus = sum(sig_minus).item();

        // Restore input
        input.data()[i] = original;

        // Finite difference approximation
        float fd_grad = (f_plus - f_minus) / (2.0f * eps);

        // Analytical gradient
        float analytical = input_grad_data[i];

        // Check match: use both absolute and relative tolerance
        float rel_error = std::fabs(fd_grad - analytical) / (std::fabs(analytical) + 1e-8f);
        float abs_error = std::fabs(fd_grad - analytical);

        // Pass if either absolute tolerance is met OR relative tolerance is met
        assert(abs_error < abs_tol || rel_error < rel_tol);
    }

    return 0;
}
