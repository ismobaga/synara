#include <cassert>
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;
    auto x = Tensor::from_vector({2}, {3, 4}, true);
    auto y = mul(x, x);
    auto s = sum(y);
    s.backward();

    assert(x.grad().at({0}) == 6.0f);
    assert(x.grad().at({1}) == 8.0f);
    return 0;
}
