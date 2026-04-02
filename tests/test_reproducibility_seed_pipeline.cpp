#include <cassert>
#include <cmath>

#include "synara/nn/dropout.hpp"
#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({1, 8}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, true);

    Tensor::manual_seed(424242);
    Dropout d1(0.5f); // auto-seeded from Tensor RNG
    Tensor y1 = d1(x);

    Tensor::manual_seed(424242);
    Dropout d2(0.5f); // should reproduce same mask/outputs
    Tensor y2 = d2(x);

    for (std::size_t i = 0; i < y1.numel(); ++i)
    {
        assert(std::fabs(y1.data()[i] - y2.data()[i]) < 1e-12);
    }

    Tensor::manual_seed(111);
    Dropout d3(0.5f);
    Tensor y3 = d3(x);

    bool any_diff = false;
    for (std::size_t i = 0; i < y1.numel(); ++i)
    {
        if (std::fabs(y1.data()[i] - y3.data()[i]) > 1e-12)
        {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);

    return 0;
}
