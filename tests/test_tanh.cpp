#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"

int main()
{
    using namespace synara;

    Tensor a = Tensor::from_vector(Shape({1, 3}), {-1.0f, 0.0f, 1.0f});
    Tensor y = tanh(a);

    constexpr float tol = 1e-6f;
    assert(std::fabs(y.at({0, 0}) - std::tanh(-1.0f)) < tol);
    assert(std::fabs(y.at({0, 1}) - 0.0f) < tol);
    assert(std::fabs(y.at({0, 2}) - std::tanh(1.0f)) < tol);

    return 0;
}
