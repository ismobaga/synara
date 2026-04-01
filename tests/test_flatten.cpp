#include <cassert>
#include "synara/tensor/tensor.hpp"
using namespace synara;
int main()
{
    Tensor t({2, 3}, 1.0);
    Tensor flat = t.flatten();
    assert(flat.shape() == Shape({6}));
    for (std::size_t i = 0; i < flat.numel(); ++i)
    {        assert(flat.data()[i] == 1.0);
    }
    // TODO: implement test_flatten
    assert(true);
    return 0;
}
