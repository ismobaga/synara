#include <cassert>
#include <cmath>
#include <vector>

#include "synara/nn/layer_norm.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    LayerNorm ln(3, true);

    Tensor x = Tensor::from_vector(Shape({2, 3}), {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    Tensor target = Tensor::from_vector(Shape({2, 3}), {
        0.2, -0.1, 0.3,
        -0.4, 0.6, -0.2,
    });

    Tensor pred = ln(x);
    Tensor loss = mse_loss(pred, target);
    loss.backward();

    Tensor &gamma = ln.weight().tensor();
    Tensor &beta = ln.bias().tensor();

    std::vector<Tensor::value_type> gamma_grad(gamma.grad().data(), gamma.grad().data() + gamma.grad().numel());
    std::vector<Tensor::value_type> beta_grad(beta.grad().data(), beta.grad().data() + beta.grad().numel());

    const Tensor::value_type eps = 1e-4;
    const Tensor::value_type abs_tol = 2e-3;
    const Tensor::value_type rel_tol = 2e-2;

    for (std::size_t i = 0; i < gamma.numel(); ++i)
    {
        const Tensor::value_type original = gamma.data()[i];

        gamma.data()[i] = original + eps;
        const Tensor::value_type plus = mse_loss(ln(x), target).item();

        gamma.data()[i] = original - eps;
        const Tensor::value_type minus = mse_loss(ln(x), target).item();

        gamma.data()[i] = original;

        const Tensor::value_type fd = (plus - minus) / (2.0 * eps);
        const Tensor::value_type an = gamma_grad[i];
        const Tensor::value_type abs_err = std::fabs(fd - an);
        const Tensor::value_type rel_err = abs_err / (std::fabs(an) + 1e-8);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    for (std::size_t i = 0; i < beta.numel(); ++i)
    {
        const Tensor::value_type original = beta.data()[i];

        beta.data()[i] = original + eps;
        const Tensor::value_type plus = mse_loss(ln(x), target).item();

        beta.data()[i] = original - eps;
        const Tensor::value_type minus = mse_loss(ln(x), target).item();

        beta.data()[i] = original;

        const Tensor::value_type fd = (plus - minus) / (2.0 * eps);
        const Tensor::value_type an = beta_grad[i];
        const Tensor::value_type abs_err = std::fabs(fd - an);
        const Tensor::value_type rel_err = abs_err / (std::fabs(an) + 1e-8);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    return 0;
}