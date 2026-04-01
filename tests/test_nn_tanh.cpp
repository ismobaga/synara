#include <cassert>
#include <cmath>

#include "synara/nn/tanh.hpp"

int main()
{
    using namespace synara;

    Tanh layer;
    Tensor x = Tensor::from_vector(Shape({2, 2}), {-1.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = layer(x);

    constexpr float tol = 1e-6f;
    assert(y.shape() == Shape({2, 2}));
    assert(std::fabs(y.at({0, 0}) - std::tanh(-1.0f)) < tol);
    assert(std::fabs(y.at({0, 1}) - 0.0f) < tol);
    assert(std::fabs(y.at({1, 0}) - std::tanh(1.0f)) < tol);
    assert(std::fabs(y.at({1, 1}) - std::tanh(2.0f)) < tol);

    return 0;
}
