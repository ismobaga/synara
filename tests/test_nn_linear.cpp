#include <cassert>

#include "synara/nn/linear.hpp"

int main()
{
    using namespace synara;

    Linear linear(3, 2, true);

    linear.weight().tensor().at({0, 0}) = 1.0f;
    linear.weight().tensor().at({0, 1}) = 2.0f;
    linear.weight().tensor().at({0, 2}) = 3.0f;
    linear.weight().tensor().at({1, 0}) = -1.0f;
    linear.weight().tensor().at({1, 1}) = 0.0f;
    linear.weight().tensor().at({1, 2}) = 1.0f;
    linear.bias().tensor().at({0, 0}) = 0.5f;
    linear.bias().tensor().at({0, 1}) = -0.5f;

    Tensor input = Tensor::from_vector(Shape({2, 3}), {1.0f, 2.0f, 3.0f,
                                                       4.0f, 5.0f, 6.0f});

    Tensor output = linear(input);

    assert(output.shape() == Shape({2, 2}));
    assert(output.at({0, 0}) == 14.5f);
    assert(output.at({0, 1}) == 1.5f);
    assert(output.at({1, 0}) == 32.5f);
    assert(output.at({1, 1}) == 1.5f);

    return 0;
}