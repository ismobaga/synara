#include <cassert>
#include <cmath>
#include <vector>

#include "synara/nn/batch_norm.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    BatchNorm1d bn(2, true);

    Tensor x = Tensor::from_vector(Shape({3, 2}), {
                                                     1.0f,
                                                     2.0f,
                                                     3.0f,
                                                     4.0f,
                                                     5.0f,
                                                     6.0f,
                                                 });

    Tensor target = Tensor::from_vector(Shape({3, 2}), {
                                                           0.5f,
                                                           -0.5f,
                                                           0.0f,
                                                           0.25f,
                                                           -0.25f,
                                                           0.75f,
                                                       });

    Tensor pred = bn(x);
    Tensor loss = mse_loss(pred, target);
    loss.backward();

    Tensor &gamma = bn.weight().tensor();
    Tensor &beta = bn.bias().tensor();

    std::vector<float> gamma_grad(gamma.grad().data(), gamma.grad().data() + gamma.grad().numel());
    std::vector<float> beta_grad(beta.grad().data(), beta.grad().data() + beta.grad().numel());

    const float eps = 1e-4f;
    const float abs_tol = 1e-3f;
    const float rel_tol = 1e-2f;

    for (std::size_t i = 0; i < gamma.numel(); ++i)
    {
        const float original = gamma.data()[i];

        gamma.data()[i] = original + eps;
        const float plus = mse_loss(bn(x), target).item();

        gamma.data()[i] = original - eps;
        const float minus = mse_loss(bn(x), target).item();

        gamma.data()[i] = original;

        const float fd = (plus - minus) / (2.0f * eps);
        const float an = gamma_grad[i];

        const float abs_err = std::fabs(fd - an);
        const float rel_err = abs_err / (std::fabs(an) + 1e-8f);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    for (std::size_t i = 0; i < beta.numel(); ++i)
    {
        const float original = beta.data()[i];

        beta.data()[i] = original + eps;
        const float plus = mse_loss(bn(x), target).item();

        beta.data()[i] = original - eps;
        const float minus = mse_loss(bn(x), target).item();

        beta.data()[i] = original;

        const float fd = (plus - minus) / (2.0f * eps);
        const float an = beta_grad[i];

        const float abs_err = std::fabs(fd - an);
        const float rel_err = abs_err / (std::fabs(an) + 1e-8f);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    return 0;
}
