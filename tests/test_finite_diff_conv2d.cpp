#include <cassert>
#include <cmath>

#include "synara/ops/convolution.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    // Input: (1, 1, 4, 4), kernel: (1, 1, 3, 3), no bias
    auto x = Tensor::from_vector(Shape({1, 1, 4, 4}),
                                 { 0.1f, -0.2f,  0.3f, -0.4f,
                                   0.5f, -0.6f,  0.7f, -0.8f,
                                   0.9f, -1.0f,  1.1f, -1.2f,
                                   1.3f, -1.4f,  1.5f, -1.6f},
                                 true);

    auto w = Tensor::from_vector(Shape({1, 1, 3, 3}),
                                 { 1.0f,  0.0f, -1.0f,
                                   2.0f,  0.0f, -2.0f,
                                   1.0f,  0.0f, -1.0f},
                                 true);

    // No bias — pass empty bias (shape {1}, value 0 but no grad)
    auto b = Tensor::from_vector(Shape({1}), {0.0f}, true);

    // Analytical gradients
    auto y = conv2d(x, w, b);
    auto s = sum(y);
    s.backward();

    std::vector<float> grad_x(x.grad().data(), x.grad().data() + x.numel());
    std::vector<float> grad_w(w.grad().data(), w.grad().data() + w.numel());
    std::vector<float> grad_b(b.grad().data(), b.grad().data() + b.numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    // Validate dL/dx
    for (std::size_t i = 0; i < x.numel(); ++i)
    {
        float orig    = x.data()[i];
        x.data()[i]   = orig + eps;
        float f_plus  = sum(conv2d(x, w, b)).item();
        x.data()[i]   = orig - eps;
        float f_minus = sum(conv2d(x, w, b)).item();
        x.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_x[i]);
        assert(err < abs_tol || err / (std::fabs(grad_x[i]) + 1e-8f) < rel_tol);
    }

    // Validate dL/dw
    for (std::size_t i = 0; i < w.numel(); ++i)
    {
        float orig    = w.data()[i];
        w.data()[i]   = orig + eps;
        float f_plus  = sum(conv2d(x, w, b)).item();
        w.data()[i]   = orig - eps;
        float f_minus = sum(conv2d(x, w, b)).item();
        w.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_w[i]);
        assert(err < abs_tol || err / (std::fabs(grad_w[i]) + 1e-8f) < rel_tol);
    }

    // Validate dL/db
    for (std::size_t i = 0; i < b.numel(); ++i)
    {
        float orig    = b.data()[i];
        b.data()[i]   = orig + eps;
        float f_plus  = sum(conv2d(x, w, b)).item();
        b.data()[i]   = orig - eps;
        float f_minus = sum(conv2d(x, w, b)).item();
        b.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_b[i]);
        assert(err < abs_tol || err / (std::fabs(grad_b[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
