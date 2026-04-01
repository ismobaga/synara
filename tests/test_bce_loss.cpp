#include <cassert>
#include <cmath>

#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    Tensor pred = Tensor::from_vector(Shape({2, 1}), {0.8f, 0.4f}, true);
    Tensor target = Tensor::from_vector(Shape({2, 1}), {1.0f, 0.0f});

    Tensor loss = binary_cross_entropy(pred, target);
    assert(loss.is_scalar());

    const float expected =
        static_cast<float>((-std::log(0.8f) - std::log(1.0f - 0.4f)) / 2.0);
    assert(std::fabs(loss.item() - expected) < 1e-6f);

    loss.backward();
    assert(pred.has_grad());

    const float g0 = pred.grad().at({0, 0});
    const float g1 = pred.grad().at({1, 0});

    const float e0 = static_cast<float>((0.8f - 1.0f) / (0.8f * 0.2f * 2.0f));
    const float e1 = static_cast<float>((0.4f - 0.0f) / (0.4f * 0.6f * 2.0f));

    assert(std::fabs(g0 - e0) < 1e-5f);
    assert(std::fabs(g1 - e1) < 1e-5f);

    return 0;
}
