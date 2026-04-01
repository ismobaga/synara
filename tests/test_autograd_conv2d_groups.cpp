#include <cassert>

#include "synara/ops/convolution.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            1, 2,
            3, 4,
            10, 20,
            30, 40,
        },
        true);

    Tensor w = Tensor::from_vector(
        Shape({2, 1, 1, 1}),
        {
            2.0f,
            -1.0f,
        },
        true);

    Tensor y = conv2d(x, w, 1, 1, 0, 0, 1, 1, 2);
    Tensor s = sum(y);
    s.backward();

    assert(x.has_grad());
    assert(w.has_grad());

    // dx is copied from corresponding group weight.
    assert(x.grad().at({0, 0, 0, 0}) == 2.0f);
    assert(x.grad().at({0, 1, 0, 0}) == -1.0f);

    // dw is sum of each group's channel values.
    assert(w.grad().at({0, 0, 0, 0}) == 10.0f);   // 1+2+3+4
    assert(w.grad().at({1, 0, 0, 0}) == 100.0f);  // 10+20+30+40

    return 0;
}
