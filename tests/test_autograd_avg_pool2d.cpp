#include <cassert>

#include "synara/ops/pooling.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 2, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
        },
        true);

    Tensor y = avg_pool2d(x, 2, 2, 1, 1, 0, 0);
    Tensor s = sum(y);
    s.backward();

    assert(y.shape() == Shape({1, 1, 1, 1}));
    assert(y.at({0, 0, 0, 0}) == 2.5f);

    assert(x.has_grad());
    assert(x.grad().shape() == x.shape());
    assert(x.grad().at({0, 0, 0, 0}) == 0.25f);
    assert(x.grad().at({0, 0, 0, 1}) == 0.25f);
    assert(x.grad().at({0, 0, 1, 0}) == 0.25f);
    assert(x.grad().at({0, 0, 1, 1}) == 0.25f);

    return 0;
}
