#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

namespace {

void test_softmax_forward()
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
}

void test_softmax_backward()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 3}), {1.0f, 2.0f, 3.0f}, true);
    Tensor y = softmax(x, -1);
    Tensor s = sum(y);
    s.backward();

    assert(x.has_grad());

    float exp1 = std::exp(1.0f);
    float exp2 = std::exp(2.0f);
    float exp3 = std::exp(3.0f);
    float sum_exp = exp1 + exp2 + exp3;

    float p1 = exp1 / sum_exp;
    float p2 = exp2 / sum_exp;
    float p3 = exp3 / sum_exp;

    constexpr float tol = 1e-5f;
    assert(std::fabs(x.grad().at({0, 0}) - (p1 * (1.0f - p1) + p2 * (-p1) + p3 * (-p1))) < tol);
    assert(std::fabs(x.grad().at({0, 1}) - (p1 * (-p2) + p2 * (1.0f - p2) + p3 * (-p2))) < tol);
    assert(std::fabs(x.grad().at({0, 2}) - (p1 * (-p3) + p2 * (-p3) + p3 * (1.0f - p3))) < tol);
}

} // namespace

int main()
{
    test_softmax_forward();
    test_softmax_backward();

    return 0;
}
