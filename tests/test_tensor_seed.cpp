#include <cassert>

#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;

    Tensor::manual_seed(123456);
    Tensor a = Tensor::uniform(Shape({128}), -1.0, 1.0);

    Tensor::manual_seed(123456);
    Tensor b = Tensor::uniform(Shape({128}), -1.0, 1.0);

    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        assert(a.data()[i] == b.data()[i]);
    }

    Tensor::manual_seed(777);
    Tensor c = Tensor::randn(Shape({64}), 0.0, 1.0);
    Tensor::manual_seed(777);
    Tensor d = Tensor::randn(Shape({64}), 0.0, 1.0);

    for (std::size_t i = 0; i < c.numel(); ++i)
    {
        assert(c.data()[i] == d.data()[i]);
    }

    Tensor::manual_seed(1);
    Tensor e = Tensor::uniform(Shape({32}), 0.0, 1.0);
    Tensor::manual_seed(2);
    Tensor f = Tensor::uniform(Shape({32}), 0.0, 1.0);

    bool any_diff = false;
    for (std::size_t i = 0; i < e.numel(); ++i)
    {
        if (e.data()[i] != f.data()[i])
        {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);

    return 0;
}
