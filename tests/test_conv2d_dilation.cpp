#include <cassert>

#include "synara/ops/convolution.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 5, 5}),
        {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
        });

    Tensor w = Tensor::from_vector(
        Shape({1, 1, 2, 2}),
        {
            1, 2,
            3, 4,
        });

    Tensor y = conv2d(x, w, 1, 1, 0, 0, 2, 2, 1);

    assert(y.shape() == Shape({1, 1, 3, 3}));
    assert(y.at({0, 0, 0, 0}) == 92.0f);
    assert(y.at({0, 0, 0, 1}) == 102.0f);
    assert(y.at({0, 0, 0, 2}) == 112.0f);

    return 0;
}
