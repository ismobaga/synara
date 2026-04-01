#include <cassert>

#include "synara/ops/activation.hpp"
int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 3}, {
        -1, 2, -3,
         4, -5, 6
    });

    auto r = relu(a);

    assert(r.at({0,0}) == 0.0f);
    assert(r.at({0,1}) == 2.0f);
    assert(r.at({0,2}) == 0.0f);
    assert(r.at({1,0}) == 4.0f);
    assert(r.at({1,1}) == 0.0f);
    assert(r.at({1,2}) == 6.0f);
    return 0;
}
