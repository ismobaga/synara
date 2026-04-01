#include <cassert>

#include "synara/ops/pooling.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 4, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
        });

    Tensor y = avg_pool2d(x, 2, 2, 2, 2, 0, 0);

    assert(y.shape() == Shape({1, 1, 2, 2}));
    assert(y.at({0, 0, 0, 0}) == 3.5f);
    assert(y.at({0, 0, 0, 1}) == 5.5f);
    assert(y.at({0, 0, 1, 0}) == 11.5f);
    assert(y.at({0, 0, 1, 1}) == 13.5f);

    return 0;
}
