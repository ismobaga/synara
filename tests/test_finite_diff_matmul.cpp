#include <cassert>
#include <cmath>

#include "synara/ops/linalg.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    auto a = Tensor::from_vector(Shape({3, 4}),
                                 {0.1f, -0.2f,  0.3f, -0.4f,
                                  0.5f, -0.6f,  0.7f, -0.8f,
                                  0.9f, -1.0f, -0.1f,  0.2f},
                                 true);

    auto b = Tensor::from_vector(Shape({4, 2}),
                                 { 0.5f, -0.5f,
                                   1.0f,  0.0f,
                                  -1.0f,  1.0f,
                                   0.2f, -0.3f},
                                 true);

    // Analytical gradients
    auto c = matmul(a, b);
    auto s = sum(c);
    s.backward();

    std::vector<float> grad_a(a.grad().data(), a.grad().data() + a.numel());
    std::vector<float> grad_b(b.grad().data(), b.grad().data() + b.numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    // Validate dL/dA
    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        float orig    = a.data()[i];
        a.data()[i]   = orig + eps;
        float f_plus  = sum(matmul(a, b)).item();
        a.data()[i]   = orig - eps;
        float f_minus = sum(matmul(a, b)).item();
        a.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_a[i]);
        assert(err < abs_tol || err / (std::fabs(grad_a[i]) + 1e-8f) < rel_tol);
    }

    // Validate dL/dB
    for (std::size_t i = 0; i < b.numel(); ++i)
    {
        float orig    = b.data()[i];
        b.data()[i]   = orig + eps;
        float f_plus  = sum(matmul(a, b)).item();
        b.data()[i]   = orig - eps;
        float f_minus = sum(matmul(a, b)).item();
        b.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_b[i]);
        assert(err < abs_tol || err / (std::fabs(grad_b[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
