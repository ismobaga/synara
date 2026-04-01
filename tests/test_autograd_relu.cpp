#include <cassert>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

int main() {
    using namespace synara;

    auto x = Tensor::from_vector({2, 3}, {
        -1, 2, -3,
         4, -5, 6
    }, true);

    auto y = relu(x);
    auto s = sum(y);
    s.backward();

    assert(x.has_grad());

    assert(x.grad().at({0,0}) == 0.0f);
    assert(x.grad().at({0,1}) == 1.0f);
    assert(x.grad().at({0,2}) == 0.0f);
    assert(x.grad().at({1,0}) == 1.0f);
    assert(x.grad().at({1,1}) == 0.0f);
    assert(x.grad().at({1,2}) == 1.0f);

    return 0;
}