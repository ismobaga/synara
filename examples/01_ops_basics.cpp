#include <iostream>

#include "synara/ops/activation.hpp"
#include "synara/ops/elementwise.hpp"
#include "synara/ops/linalg.hpp"
#include "synara/ops/reduction.hpp"

int main() {
    using namespace synara;

    auto a = Tensor::from_vector({2, 2}, {1, 2, 3, 4});
    auto b = Tensor::from_vector({2, 2}, {10, 20, 30, 40});

    auto c = add(a, b);
    auto d = mul(a, b);
    auto s = sum(a);
    auto m = mean(a);

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "a + b = " << c << "\n";
    std::cout << "a * b = " << d << "\n";
    std::cout << "sum(a) = " << s << "\n";
    std::cout << "mean(a) = " << m << "\n\n";

    auto x = Tensor::from_vector({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    auto y = Tensor::from_vector({3, 2}, {
        7, 8,
        9, 10,
        11, 12
    });

    auto z = matmul(x, y);

    std::cout << "x = " << x << "\n";
    std::cout << "y = " << y << "\n";
    std::cout << "x @ y = " << z << "\n\n";

    auto q = Tensor::from_vector({2, 3}, {
        -1, 2, -3,
         4, -5, 6
    });

    std::cout << "q = " << q << "\n";
    std::cout << "relu(q) = " << relu(q) << "\n";

    return 0;
}