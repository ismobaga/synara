#include <cassert>
#include "synara/ops/linalg.hpp"

int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 3}, {1, 2, 3,
                                          4, 5, 6});

    auto b = Tensor::from_vector({3, 2}, {7, 8,
                                          9, 10,
                                          11, 12});

    auto c = matmul(a, b);

    assert(c.shape() == Shape({2, 2}));
    assert(c.at({0, 0}) == 58.0f);
    assert(c.at({0, 1}) == 64.0f);
    assert(c.at({1, 0}) == 139.0f);
    assert(c.at({1, 1}) == 154.0f);

    auto a_base = Tensor::from_vector({3, 2}, {1, 4,
                                               2, 5,
                                               3, 6});
    auto b_base = Tensor::from_vector({2, 3}, {7, 9, 11,
                                               8, 10, 12});

    auto a_view = a_base.transpose(0, 1);
    auto b_view = b_base.transpose(0, 1);
    assert(!a_view.is_contiguous());
    assert(!b_view.is_contiguous());

    auto c_view = matmul(a_view, b_view);
    assert(c_view.shape() == Shape({2, 2}));
    assert(c_view.at({0, 0}) == 58.0f);
    assert(c_view.at({0, 1}) == 64.0f);
    assert(c_view.at({1, 0}) == 139.0f);
    assert(c_view.at({1, 1}) == 154.0f);
    return 0;
}
