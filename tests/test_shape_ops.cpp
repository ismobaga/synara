#include <cassert>

#include "synara/ops/shape.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor a = Tensor::from_vector(Shape({2, 2}), {1, 2, 3, 4}, true);
    Tensor b = Tensor::from_vector(Shape({2, 2}), {5, 6, 7, 8}, true);

    Tensor c = cat({a, b}, 0);
    assert(c.shape() == Shape({4, 2}));
    assert(c.data()[0] == 1.0f);
    assert(c.data()[6] == 7.0f);

    Tensor s = stack({a, b}, 0);
    assert(s.shape() == Shape({2, 2, 2}));
    assert(s.data()[0] == 1.0f);
    assert(s.data()[7] == 8.0f);

    auto parts = split(c, 2, 0);
    assert(parts.size() == 2);
    assert(parts[0].shape() == Shape({2, 2}));
    assert(parts[1].shape() == Shape({2, 2}));
    assert(parts[1].data()[0] == 5.0f);

    Tensor q = a.unsqueeze(0).permute({1, 0, 2}).squeeze(1);
    Tensor loss = sum(q);
    loss.backward();
    assert(a.has_grad());
    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        assert(a.grad().data()[i] == 1.0f);
    }

    return 0;
}
