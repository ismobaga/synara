#include <cassert>
#include <algorithm>

#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;

    Tensor t = Tensor::uniform(Shape({100, 100}), 0.0, 1.0);
    assert(t.shape() == Shape({100, 100}));
    assert(t.numel() == 10000);

    Tensor::value_type min_val = t.data()[0];
    Tensor::value_type max_val = t.data()[0];
    for (std::size_t i = 0; i < t.numel(); ++i)
    {
        min_val = std::min(min_val, t.data()[i]);
        max_val = std::max(max_val, t.data()[i]);
    }

    assert(min_val >= 0.0);
    assert(max_val <= 1.0);

    Tensor t2 = Tensor::uniform(Shape({50}), -5.0, 5.0);
    for (std::size_t i = 0; i < t2.numel(); ++i)
    {
        assert(t2.data()[i] >= -5.0);
        assert(t2.data()[i] <= 5.0);
    }

    return 0;
}
