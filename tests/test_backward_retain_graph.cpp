#include <cassert>

#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1}), {3.0f}, true);
    Tensor y = mul(x, x);
    Tensor loss = sum(y);

    loss.backward(true);
    assert(x.has_grad());
    assert(x.grad().data()[0] == 6.0f);

    loss.backward(true);
    assert(x.grad().data()[0] == 12.0f);

    x.zero_grad();
    loss.backward(false);
    assert(x.grad().data()[0] == 6.0f);

    x.zero_grad();
    loss.backward(false);
    assert(x.grad().data()[0] == 0.0f);

    return 0;
}
