#include <cassert>

#include "synara/ops/convolution.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 3, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
        },
        true);

    Tensor w = Tensor::from_vector(
        Shape({1, 1, 2, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
        },
        true);

    Tensor b = Tensor::from_vector(Shape({1}), {0.5f}, true);

    Tensor y = conv2d(x, w, b);
    Tensor s = sum(y);
    s.backward();

    assert(x.has_grad());
    assert(w.has_grad());
    assert(b.has_grad());

    assert(x.grad().shape() == x.shape());
    assert(w.grad().shape() == w.shape());
    assert(b.grad().shape() == b.shape());

    // d(sum(y))/dx for this setup.
    assert(x.grad().at({0, 0, 0, 0}) == 1.0f);
    assert(x.grad().at({0, 0, 0, 1}) == 3.0f);
    assert(x.grad().at({0, 0, 0, 2}) == 2.0f);
    assert(x.grad().at({0, 0, 1, 0}) == 4.0f);
    assert(x.grad().at({0, 0, 1, 1}) == 10.0f);
    assert(x.grad().at({0, 0, 1, 2}) == 6.0f);
    assert(x.grad().at({0, 0, 2, 0}) == 3.0f);
    assert(x.grad().at({0, 0, 2, 1}) == 7.0f);
    assert(x.grad().at({0, 0, 2, 2}) == 4.0f);

    // d(sum(y))/dw for this setup.
    assert(w.grad().at({0, 0, 0, 0}) == 12.0f);
    assert(w.grad().at({0, 0, 0, 1}) == 16.0f);
    assert(w.grad().at({0, 0, 1, 0}) == 24.0f);
    assert(w.grad().at({0, 0, 1, 1}) == 28.0f);

    // d(sum(y))/db = number of output elements per channel.
    assert(b.grad().at({0}) == 4.0f);

    return 0;
}
