#include <cassert>
#include <cmath>
#include <vector>

#include "synara/nn/batch_norm.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    BatchNorm2d bn(2, true);

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
            10.0f, 20.0f,
            30.0f, 40.0f,
        });

    Tensor target = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            0.2f, -0.1f,
            0.3f, -0.4f,
            -0.5f, 0.6f,
            -0.7f, 0.8f,
        });

    Tensor pred = bn(x);
    Tensor loss = mse_loss(pred, target);
    loss.backward();

    Tensor &gamma = bn.weight().tensor();
    Tensor &beta = bn.bias().tensor();

    std::vector<double> gamma_grad(gamma.grad().data(), gamma.grad().data() + gamma.grad().numel());
    std::vector<double> beta_grad(beta.grad().data(), beta.grad().data() + beta.grad().numel());

    const double eps = 1e-4;
    const double abs_tol = 1e-3;
    const double rel_tol = 1e-2;

    for (std::size_t i = 0; i < gamma.numel(); ++i)
    {
        const double original = gamma.data()[i];

        gamma.data()[i] = original + eps;
        const double plus = mse_loss(bn(x), target).item();

        gamma.data()[i] = original - eps;
        const double minus = mse_loss(bn(x), target).item();

        gamma.data()[i] = original;

        const double fd = (plus - minus) / (2.0 * eps);
        const double an = gamma_grad[i];

        const double abs_err = std::fabs(fd - an);
        const double rel_err = abs_err / (std::fabs(an) + 1e-8);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    for (std::size_t i = 0; i < beta.numel(); ++i)
    {
        const double original = beta.data()[i];

        beta.data()[i] = original + eps;
        const double plus = mse_loss(bn(x), target).item();

        beta.data()[i] = original - eps;
        const double minus = mse_loss(bn(x), target).item();

        beta.data()[i] = original;

        const double fd = (plus - minus) / (2.0 * eps);
        const double an = beta_grad[i];

        const double abs_err = std::fabs(fd - an);
        const double rel_err = abs_err / (std::fabs(an) + 1e-8);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    return 0;
}
