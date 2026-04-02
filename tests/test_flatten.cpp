#include <cassert>
#include "synara/tensor/tensor.hpp"
using namespace synara;
int main()
{
    // 1. Flatten a 2-D tensor of ones
    Tensor t({2, 3}, 1.0);
    Tensor flat = t.flatten();
    assert(flat.shape() == Shape({6}));
    for (std::size_t i = 0; i < flat.numel(); ++i)
    {
        assert(flat.data()[i] == 1.0);
    }

    // 2. Flatten a 1-D tensor — should be a no-op in shape
    Tensor v = Tensor::from_vector(Shape({5}), {1, 2, 3, 4, 5});
    Tensor vf = v.flatten();
    assert(vf.shape() == Shape({5}));
    for (std::size_t i = 0; i < 5; ++i)
        assert(vf.data()[i] == static_cast<float>(i + 1));

    // 3. Flatten a 3-D tensor — shape should collapse to 1-D
    Tensor t3({2, 3, 4}, 2.0f);
    Tensor flat3 = t3.flatten();
    assert(flat3.shape() == Shape({24}));
    for (std::size_t i = 0; i < flat3.numel(); ++i)
        assert(flat3.data()[i] == 2.0f);

    // 4. Flatten a 4-D tensor
    Tensor t4({2, 2, 2, 2}, 3.0f);
    Tensor flat4 = t4.flatten();
    assert(flat4.shape() == Shape({16}));
    for (std::size_t i = 0; i < flat4.numel(); ++i)
        assert(flat4.data()[i] == 3.0f);

    // 5. Values are preserved correctly across a non-trivial reshape
    Tensor vals = Tensor::from_vector(Shape({2, 3}), {10, 20, 30, 40, 50, 60});
    Tensor vflat = vals.flatten();
    assert(vflat.shape() == Shape({6}));
    assert(vflat.data()[0] == 10.0f);
    assert(vflat.data()[1] == 20.0f);
    assert(vflat.data()[2] == 30.0f);
    assert(vflat.data()[3] == 40.0f);
    assert(vflat.data()[4] == 50.0f);
    assert(vflat.data()[5] == 60.0f);

    return 0;
}
