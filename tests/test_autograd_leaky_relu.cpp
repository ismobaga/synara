#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 4}), {-2.0f, -1.0f, 0.0f, 3.0f}, true);
    Tensor y = leaky_relu(x, 0.1f);
    Tensor s = sum(y);
    s.backward();

    assert(x.has_grad());

    constexpr float tol = 1e-6f;
    assert(std::fabs(x.grad().at({0, 0}) - 0.1f) < tol);
    assert(std::fabs(x.grad().at({0, 1}) - 0.1f) < tol);
    assert(std::fabs(x.grad().at({0, 2}) - 0.1f) < tol);
    assert(std::fabs(x.grad().at({0, 3}) - 1.0f) < tol);

    return 0;
}
