#include <cassert>
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 2}, {1, 2, 3, 4});

    auto s = sum(a);
    auto m = mean(a);

    assert(s.shape() == Shape({}));
    assert(m.shape() == Shape({}));

    assert(s.item() == 10.0f);
    assert(m.item() == 2.5f);
    return 0;
}
