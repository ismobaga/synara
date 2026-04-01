#include <cassert>
#include "synara/ops/elementwise.hpp"


int main()
{
    using namespace synara;

    auto a = Tensor::from_vector({2, 2}, {1, 2, 3, 4});
    auto b = Tensor::from_vector({2, 2}, {10, 20, 30, 40});

    auto c = add(a, b);
    assert(c.at({0,0}) == 11.0f);
    assert(c.at({1,1}) == 44.0f);

    auto d = sub(b, a);
    assert(d.at({0,0}) == 9.0f);
    assert(d.at({1,1}) == 36.0f);

    auto e = mul(a, b);
    assert(e.at({0,0}) == 10.0f);
    assert(e.at({1,1}) == 160.0f);

    auto f = div(b, a);
    assert(f.at({0,0}) == 10.0f);
    assert(f.at({1,1}) == 10.0f);

    auto g = add(a, 5.0f);
    assert(g.at({0,0}) == 6.0f);
    assert(g.at({1,1}) == 9.0f);

    auto h = mul(2.0f, a);
    assert(h.at({0,0}) == 2.0f);
    assert(h.at({1,1}) == 8.0f);
    return 0;
}
