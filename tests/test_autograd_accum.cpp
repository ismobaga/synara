#include <cassert>
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"
int main()
{
    using namespace synara;
    auto x = Tensor::from_vector({2}, {3, 4}, true);
    auto a = mul(x, x);
    auto b = add(a, x);
    auto s = sum(b);
    s.backward();

    assert(x.grad().at({0}) == 7.0f);
    assert(x.grad().at({1}) == 9.0f);

    // d/dx (x^2 + x) = 2x + 1
    // => {7, 9}
    return 0;
}
