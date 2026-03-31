#include <iostream>

#include "synara/tensor/tensor.hpp"

int main()
{
    using synara::Shape;
    using synara::Slice;
    using synara::Tensor;

    Tensor a = Tensor::from_vector(Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = a.transpose(0, 1);
    Tensor c = a.reshape(Shape{3, 2});
    Tensor d = a.slice(1, Slice{1, 3, 1});

    std::cout << "a(1, 2) = " << a({1, 2}) << "\n";
    std::cout << "b shape = " << b.shape().to_string() << "\n";
    std::cout << "c shape = " << c.shape().to_string() << "\n";
    std::cout << "d shape = " << d.shape().to_string() << "\n";
    std::cout << "d(0, 0) = " << d({0, 0}) << "\n";

    return 0;
}
