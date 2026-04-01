#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 3}), {1.0f, 2.0f, 3.0f});
    Tensor y = softmax(x, -1);

    constexpr float tol = 1e-6f;
    float exp1 = std::exp(1.0f);
    float exp2 = std::exp(2.0f);
    float exp3 = std::exp(3.0f);
    float sum_exp = exp1 + exp2 + exp3;

    assert(std::fabs(y.at({0, 0}) - (exp1 / sum_exp)) < tol);
    assert(std::fabs(y.at({0, 1}) - (exp2 / sum_exp)) < tol);
    assert(std::fabs(y.at({0, 2}) - (exp3 / sum_exp)) < tol);

    float sum_probs = y.at({0, 0}) + y.at({0, 1}) + y.at({0, 2});
    assert(std::fabs(sum_probs - 1.0f) < tol);

    return 0;
}
