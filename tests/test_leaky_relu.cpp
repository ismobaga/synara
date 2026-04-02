#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

namespace {

void test_leaky_relu_forward()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 4}), {-2.0f, -1.0f, 0.0f, 3.0f});
    Tensor y = leaky_relu(x, 0.1f);

    constexpr float tol = 1e-6f;
    assert(std::fabs(y.at({0, 0}) - (-0.2f)) < tol);
    assert(std::fabs(y.at({0, 1}) - (-0.1f)) < tol);
    assert(std::fabs(y.at({0, 2}) - 0.0f) < tol);
    assert(std::fabs(y.at({0, 3}) - 3.0f) < tol);
}

void test_leaky_relu_backward()
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
}

} // namespace

int main()
{
    test_leaky_relu_forward();
    test_leaky_relu_backward();

    return 0;
}
