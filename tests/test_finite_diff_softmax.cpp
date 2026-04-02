#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    // Input: (2, 4)
    auto input = Tensor::from_vector(Shape({2, 4}),
                                     { 0.5f, -1.0f,  2.0f, -0.5f,
                                      -0.3f,  1.5f, -2.0f,  0.8f},
                                     true);

    // Analytical gradients: sum(softmax(x)) w.r.t. x
    auto out = softmax(input, -1);
    auto s   = sum(out);
    s.backward();

    std::vector<float> grad(input.grad().data(),
                            input.grad().data() + input.numel());

    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    for (std::size_t i = 0; i < input.numel(); ++i)
    {
        float orig    = input.data()[i];
        input.data()[i] = orig + eps;
        float f_plus  = sum(softmax(input, -1)).item();
        input.data()[i] = orig - eps;
        float f_minus = sum(softmax(input, -1)).item();
        input.data()[i] = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad[i]);
        assert(err < abs_tol || err / (std::fabs(grad[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
