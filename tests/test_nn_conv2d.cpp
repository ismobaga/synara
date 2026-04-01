#include <cassert>

#include "synara/nn/conv2d.hpp"

int main()
{
    using namespace synara;

    Conv2d conv(1, 1, 2, 2, 1, 1, 0, 0, true);

    conv.weight().tensor().at({0, 0, 0, 0}) = 1.0f;
    conv.weight().tensor().at({0, 0, 0, 1}) = 0.0f;
    conv.weight().tensor().at({0, 0, 1, 0}) = 0.0f;
    conv.weight().tensor().at({0, 0, 1, 1}) = 1.0f;
    conv.bias().tensor().at({0}) = 1.0f;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 3, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
        });

    Tensor y = conv(x);

    assert(conv.parameters().size() == 2);
    assert(y.shape() == Shape({1, 1, 2, 2}));
    assert(y.at({0, 0, 0, 0}) == 7.0f);
    assert(y.at({0, 0, 0, 1}) == 9.0f);
    assert(y.at({0, 0, 1, 0}) == 13.0f);
    assert(y.at({0, 0, 1, 1}) == 15.0f);

    return 0;
}
