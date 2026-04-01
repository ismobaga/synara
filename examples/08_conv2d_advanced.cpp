#include <iostream>

#include "synara/nn/conv2d.hpp"

int main()
{
    using namespace synara;

    Conv2d grouped(2, 2, 1, 1, 1, 1, 0, 0, true, 1, 1, 2);

    grouped.weight().tensor().at({0, 0, 0, 0}) = 2.0f;
    grouped.weight().tensor().at({1, 0, 0, 0}) = -1.0f;
    grouped.bias().tensor().at({0}) = 0.0f;
    grouped.bias().tensor().at({1}) = 1.0f;

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            1, 2,
            3, 4,
            10, 20,
            30, 40,
        });

    Tensor y = grouped(x);

    std::cout << "grouped conv output:\n" << y << "\n\n";

    Conv2d dilated(1, 1, 2, 2, 1, 1, 0, 0, false, 2, 2, 1);
    dilated.weight().tensor().at({0, 0, 0, 0}) = 1.0f;
    dilated.weight().tensor().at({0, 0, 0, 1}) = 2.0f;
    dilated.weight().tensor().at({0, 0, 1, 0}) = 3.0f;
    dilated.weight().tensor().at({0, 0, 1, 1}) = 4.0f;

    Tensor x2 = Tensor::from_vector(
        Shape({1, 1, 5, 5}),
        {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
        });

    Tensor y2 = dilated(x2);
    std::cout << "dilated conv output:\n" << y2 << "\n";

    return 0;
}
