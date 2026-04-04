#include <cassert>
#include <cmath>

#include "synara/ops/linalg.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 3}, {1, 2, 3, 4, 5, 6}, true);

    auto b = Tensor::from_vector({3, 2}, {7, 8, 9, 10, 11, 12}, true);

    auto c = matmul(a, b);
    auto loss = sum(c);
    loss.backward();

    assert(c.shape() == Shape({2, 2}));
    assert(std::abs(c.at({0, 0}) - 58.0) < 1e-9);
    assert(std::abs(c.at({0, 1}) - 64.0) < 1e-9);
    assert(std::abs(c.at({1, 0}) - 139.0) < 1e-9);
    assert(std::abs(c.at({1, 1}) - 154.0) < 1e-9);

    const auto &grad_a = a.grad();
    const auto &grad_b = b.grad();

    assert(grad_a.shape() == Shape({2, 3}));
    assert(grad_b.shape() == Shape({3, 2}));

    assert(std::abs(grad_a.at({0, 0}) - 15.0) < 1e-9);
    assert(std::abs(grad_a.at({0, 1}) - 19.0) < 1e-9);
    assert(std::abs(grad_a.at({0, 2}) - 23.0) < 1e-9);
    assert(std::abs(grad_a.at({1, 0}) - 15.0) < 1e-9);
    assert(std::abs(grad_a.at({1, 1}) - 19.0) < 1e-9);
    assert(std::abs(grad_a.at({1, 2}) - 23.0) < 1e-9);

    assert(std::abs(grad_b.at({0, 0}) - 5.0) < 1e-9);
    assert(std::abs(grad_b.at({1, 0}) - 7.0) < 1e-9);
    assert(std::abs(grad_b.at({2, 0}) - 9.0) < 1e-9);
    assert(std::abs(grad_b.at({0, 1}) - 5.0) < 1e-9);
    assert(std::abs(grad_b.at({1, 1}) - 7.0) < 1e-9);
    assert(std::abs(grad_b.at({2, 1}) - 9.0) < 1e-9);

    return 0;
}
