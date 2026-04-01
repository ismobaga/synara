#include <cassert>

#include "synara/nn/conv2d.hpp"

int main()
{
    using namespace synara;

    Conv2d conv(2, 2, 1, 1, 1, 1, 0, 0, true, 1, 1, 2);

    conv.weight().tensor().at({0, 0, 0, 0}) = 3.0f;
    conv.weight().tensor().at({1, 0, 0, 0}) = -2.0f;
    conv.bias().tensor().at({0}) = 1.0f;
    conv.bias().tensor().at({1}) = 0.0f;

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 1, 2}),
        {
            2.0f, 4.0f,
            5.0f, 6.0f,
        });

    Tensor y = conv(x);

    assert(y.shape() == Shape({1, 2, 1, 2}));
    assert(y.at({0, 0, 0, 0}) == 7.0f);
    assert(y.at({0, 0, 0, 1}) == 13.0f);
    assert(y.at({0, 1, 0, 0}) == -10.0f);
    assert(y.at({0, 1, 0, 1}) == -12.0f);

    return 0;
}
