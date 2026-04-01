#include <cassert>

#include "synara/ops/convolution.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            // channel 0
            1, 2,
            3, 4,
            // channel 1
            10, 20,
            30, 40,
        });

    // groups=2: each output channel uses only one input channel.
    Tensor w = Tensor::from_vector(
        Shape({2, 1, 1, 1}),
        {
            2.0f,
            -1.0f,
        });

    Tensor y = conv2d(x, w, 1, 1, 0, 0, 1, 1, 2);

    assert(y.shape() == Shape({1, 2, 2, 2}));

    assert(y.at({0, 0, 0, 0}) == 2.0f);
    assert(y.at({0, 0, 1, 1}) == 8.0f);

    assert(y.at({0, 1, 0, 0}) == -10.0f);
    assert(y.at({0, 1, 1, 1}) == -40.0f);

    return 0;
}
