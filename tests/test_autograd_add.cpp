#include <cassert>
// #include "synara/ops/activation.hpp"
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;
    auto x = Tensor::from_vector({2}, {1, 2}, true);
    auto y = Tensor::from_vector({2}, {10, 20}, true);

    auto z = add(x, y);
    auto s = sum(z);
    s.backward();

    assert(x.grad().at({0}) == 1.0f);
    assert(x.grad().at({1}) == 1.0f);
    assert(y.grad().at({0}) == 1.0f);
    assert(y.grad().at({1}) == 1.0f);
    return 0;
}
