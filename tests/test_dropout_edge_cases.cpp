#include <cassert>
#include <cmath>

#include "synara/nn/dropout.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 4}), {1.0f, -2.0f, 3.0f, -4.0f}, false);

    Dropout d0(0.0f, 7ULL);
    Tensor y0 = d0(x);
    for (Size i = 0; i < x.numel(); ++i)
    {
        assert(std::fabs(y0.data()[i] - x.data()[i]) < 1e-7f);
    }

    Dropout d1(1.0f, 7ULL);
    Tensor y1 = d1(x);
    for (Size i = 0; i < x.numel(); ++i)
    {
        assert(std::fabs(y1.data()[i]) < 1e-7f);
    }

    return 0;
}
