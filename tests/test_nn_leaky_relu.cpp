#include <cassert>
#include <cmath>

#include "synara/nn/leaky_relu.hpp"

int main()
{
    using namespace synara;

    LeakyReLU layer(0.2f);
    Tensor x = Tensor::from_vector(Shape({2, 2}), {-1.0f, 2.0f, -3.0f, 4.0f});
    Tensor y = layer(x);

    constexpr float tol = 1e-6f;
    assert(y.shape() == Shape({2, 2}));
    assert(std::fabs(y.at({0, 0}) - (-0.2f)) < tol);
    assert(std::fabs(y.at({0, 1}) - 2.0f) < tol);
    assert(std::fabs(y.at({1, 0}) - (-0.6f)) < tol);
    assert(std::fabs(y.at({1, 1}) - 4.0f) < tol);

    return 0;
}
