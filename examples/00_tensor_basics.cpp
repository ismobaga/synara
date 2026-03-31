#include <iostream>

#include "synara/tensor/tensor.hpp"

int main() {
    using namespace synara;

    auto x = Tensor::from_vector(Shape({2, 3}), {
        1, 2, 3,
        4, 5, 6
    });

    auto y = x.reshape(Shape({3, 2}));
    auto z = x.transpose(0, 1);

    std::cout << "x = " << x << "\n";
    std::cout << "y = " << y << "\n";
    std::cout << "z = " << z << "\n\n";

    x.at({1, 2}) = 99.0f;

    std::cout << "after mutation:\n";
    std::cout << "x = " << x << "\n";
    std::cout << "y = " << y << "\n";
    std::cout << "z = " << z << "\n";

    return 0;
}