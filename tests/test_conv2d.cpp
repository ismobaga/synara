#include <cassert>

#include "synara/ops/convolution.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 3, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
        });

    Tensor w = Tensor::from_vector(
        Shape({1, 1, 2, 2}),
        {
            1.0f, 0.0f,
            0.0f, 1.0f,
        });

    Tensor b = Tensor::from_vector(Shape({1}), {1.0f});

    Tensor y = conv2d(x, w, b, 1, 1, 0, 0);

    assert(y.shape() == Shape({1, 1, 2, 2}));

    assert(y.at({0, 0, 0, 0}) == 7.0f);
    assert(y.at({0, 0, 0, 1}) == 9.0f);
    assert(y.at({0, 0, 1, 0}) == 13.0f);
    assert(y.at({0, 0, 1, 1}) == 15.0f);

    Tensor y_pad = conv2d(x, w, 1, 1, 1, 1);
    assert(y_pad.shape() == Shape({1, 1, 4, 4}));

    return 0;
}
