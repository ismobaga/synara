#include <cassert>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

namespace {

void test_relu_forward()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 3}, {
        -1, 2, -3,
         4, -5, 6
    });

    auto r = relu(a);

    assert(r.at({0, 0}) == 0.0f);
    assert(r.at({0, 1}) == 2.0f);
    assert(r.at({0, 2}) == 0.0f);
    assert(r.at({1, 0}) == 4.0f);
    assert(r.at({1, 1}) == 0.0f);
    assert(r.at({1, 2}) == 6.0f);
}

void test_relu_backward()
{
    using namespace synara;

    auto x = Tensor::from_vector({2, 3}, {
        -1, 2, -3,
         4, -5, 6
    }, true);

    auto y = relu(x);
    auto s = sum(y);
    s.backward();

    assert(x.has_grad());
    assert(x.grad().at({0, 0}) == 0.0f);
    assert(x.grad().at({0, 1}) == 1.0f);
    assert(x.grad().at({0, 2}) == 0.0f);
    assert(x.grad().at({1, 0}) == 1.0f);
    assert(x.grad().at({1, 1}) == 0.0f);
    assert(x.grad().at({1, 2}) == 1.0f);
}

} // namespace

int main()
{
    test_relu_forward();
    test_relu_backward();
    return 0;
}
