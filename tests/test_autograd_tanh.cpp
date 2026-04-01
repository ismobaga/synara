#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 3}), {-1.0f, 0.0f, 1.0f}, true);
    Tensor y = tanh(x);
    Tensor s = sum(y);
    s.backward();

    assert(x.has_grad());

    constexpr float tol = 1e-5f;
    const float t0 = std::tanh(-1.0f);
    const float t1 = std::tanh(0.0f);
    const float t2 = std::tanh(1.0f);

    assert(std::fabs(x.grad().at({0, 0}) - (1.0f - t0 * t0)) < tol);
    assert(std::fabs(x.grad().at({0, 1}) - (1.0f - t1 * t1)) < tol);
    assert(std::fabs(x.grad().at({0, 2}) - (1.0f - t2 * t2)) < tol);

    return 0;
}
