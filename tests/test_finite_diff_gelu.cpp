#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    auto input = Tensor::from_vector(Shape({3, 2}),
                                     {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                                     true);

    // Analytical gradients
    auto output = gelu(input);
    auto s      = sum(output);
    s.backward();

    std::vector<float> analytical(input.grad().data(),
                                  input.grad().data() + input.grad().numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    for (std::size_t i = 0; i < input.numel(); ++i)
    {
        const float orig = input.data()[i];

        input.data()[i] = orig + eps;
        float f_plus = sum(gelu(input)).item();

        input.data()[i] = orig - eps;
        float f_minus = sum(gelu(input)).item();

        input.data()[i] = orig;

        const float fd_grad  = (f_plus - f_minus) / (2.0f * eps);
        const float abs_err  = std::fabs(fd_grad - analytical[i]);
        const float rel_err  = abs_err / (std::fabs(analytical[i]) + 1e-8f);

        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    return 0;
}
