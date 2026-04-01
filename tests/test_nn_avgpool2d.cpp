#include <cassert>

#include "synara/nn/avgpool2d.hpp"

int main()
{
    using namespace synara;

    AvgPool2d pool(2, 2, 2, 2, 0, 0);

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 4, 4}),
        {
            0.0f, 2.0f, 1.0f, 3.0f,
            4.0f, 6.0f, 5.0f, 7.0f,
            8.0f, 10.0f, 9.0f, 11.0f,
            12.0f, 14.0f, 13.0f, 15.0f,
        });

    Tensor y = pool(x);

    assert(pool.parameters().empty());
    assert(y.shape() == Shape({1, 1, 2, 2}));
    assert(y.at({0, 0, 0, 0}) == 3.0f);
    assert(y.at({0, 0, 0, 1}) == 4.0f);
    assert(y.at({0, 0, 1, 0}) == 11.0f);
    assert(y.at({0, 0, 1, 1}) == 12.0f);

    return 0;
}
