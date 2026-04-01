#include <iostream>

#include "synara/nn/conv2d.hpp"
#include "synara/nn/relu.hpp"

int main()
{
    using namespace synara;

    auto print_shape = [](const char *name, const Tensor &t) {
        std::cout << name << "(";
        for (Size i = 0; i < t.rank(); ++i)
        {
            std::cout << t.shape()[i];
            if (i + 1 < t.rank())
            {
                std::cout << ", ";
            }
        }
        std::cout << ")\n";
    };

    Conv2d conv(1, 2, 3, 3, 1, 1, 1, 1, true);
    ReLU relu;

    // Simple deterministic setup for demonstration.
    conv.weight().tensor().at({0, 0, 0, 0}) = 0.2f;
    conv.weight().tensor().at({0, 0, 1, 1}) = 0.5f;
    conv.weight().tensor().at({0, 0, 2, 2}) = 0.2f;
    conv.weight().tensor().at({1, 0, 0, 2}) = -0.3f;
    conv.weight().tensor().at({1, 0, 1, 1}) = 0.7f;
    conv.weight().tensor().at({1, 0, 2, 0}) = -0.3f;
    conv.bias().tensor().at({0}) = 0.1f;
    conv.bias().tensor().at({1}) = -0.1f;

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 4, 4}),
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
        });

    Tensor y = conv(x);
    Tensor z = relu(y);

    std::cout << "shapes:\n";
    print_shape("  input: ", x);
    print_shape("  conv:  ", y);
    print_shape("  relu:  ", z);
    std::cout << "\n";

    std::cout << "conv output:\n" << y << "\n\n";
    std::cout << "relu output:\n" << z << "\n";

    return 0;
}
