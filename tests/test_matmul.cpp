#include <cassert>
#include "synara/ops/linalg.hpp"

int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    auto b = Tensor::from_vector({3, 2}, {
        7, 8,
        9, 10,
        11, 12
    });

    auto c = matmul(a, b);

    assert(c.shape() == Shape({2, 2}));
    assert(c.at({0,0}) == 58.0f);
    assert(c.at({0,1}) == 64.0f);
    assert(c.at({1,0}) == 139.0f);
    assert(c.at({1,1}) == 154.0f);
    return 0;
}
