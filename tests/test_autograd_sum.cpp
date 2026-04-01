#include <cassert>
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;
    auto x = Tensor::from_vector({2, 2}, {1, 2, 3, 4}, true);
    auto s = sum(x);
    s.backward();

    assert(x.grad().at({0, 0}) == 1.0f);
    assert(x.grad().at({0, 1}) == 1.0f);
    assert(x.grad().at({1, 0}) == 1.0f);
    assert(x.grad().at({1, 1}) == 1.0f);
    return 0;
}
