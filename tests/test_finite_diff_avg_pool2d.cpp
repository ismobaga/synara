#include <cassert>
#include <cmath>

#include "synara/ops/pooling.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    // Input: (1, 2, 4, 4), pool 2x2 stride 2 => output (1, 2, 2, 2)
    auto x = Tensor::from_vector(Shape({1, 2, 4, 4}),
                                 {  1.0f,  2.0f,  3.0f,  4.0f,
                                    5.0f,  6.0f,  7.0f,  8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f,
                                   // channel 1
                                   -1.0f, -2.0f, -3.0f, -4.0f,
                                   -5.0f, -6.0f, -7.0f, -8.0f,
                                   -9.0f,-10.0f,-11.0f,-12.0f,
                                  -13.0f,-14.0f,-15.0f,-16.0f},
                                 true);

    // Analytical gradients
    auto y = avg_pool2d(x, 2, 2, 2, 2, 0, 0);
    auto s = sum(y);
    s.backward();

    std::vector<float> grad(x.grad().data(), x.grad().data() + x.numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    for (std::size_t i = 0; i < x.numel(); ++i)
    {
        float orig    = x.data()[i];
        x.data()[i]   = orig + eps;
        float f_plus  = sum(avg_pool2d(x, 2, 2, 2, 2, 0, 0)).item();
        x.data()[i]   = orig - eps;
        float f_minus = sum(avg_pool2d(x, 2, 2, 2, 2, 0, 0)).item();
        x.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad[i]);
        assert(err < abs_tol || err / (std::fabs(grad[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
