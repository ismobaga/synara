#include <iostream>

#include "synara/nn/maxpool2d.hpp"

int main()
{
    using namespace synara;

    MaxPool2d pool(2, 2, 2, 2, 0, 0);

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 4, 4}),
        {
            1.0f, 5.0f, 2.0f, 3.0f,
            4.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
        });

    Tensor y = pool(x);

    std::cout << "input:\n" << x << "\n\n";
    std::cout << "max_pool2d output:\n" << y << "\n";

    return 0;
}
