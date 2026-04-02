#include <cassert>
#include <cmath>

#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    // Logits: (2, 4) — two samples, four classes
    auto logits = Tensor::from_vector(Shape({2, 4}),
                                      { 0.5f, -0.3f,  1.2f, -0.8f,
                                       -1.0f,  2.0f,  0.1f,  0.4f},
                                      true);

    // One-hot-like soft targets
    auto targets = Tensor::from_vector(Shape({2, 4}),
                                       {0.0f, 0.0f, 1.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f, 0.0f});

    // Analytical gradients
    auto loss = cross_entropy_loss(logits, targets);
    loss.backward();

    std::vector<float> grad(logits.grad().data(),
                            logits.grad().data() + logits.numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 5e-2f;

    for (std::size_t i = 0; i < logits.numel(); ++i)
    {
        float orig    = logits.data()[i];
        logits.data()[i] = orig + eps;
        float f_plus  = cross_entropy_loss(logits, targets).item();
        logits.data()[i] = orig - eps;
        float f_minus = cross_entropy_loss(logits, targets).item();
        logits.data()[i] = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad[i]);
        assert(err < abs_tol || err / (std::fabs(grad[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
