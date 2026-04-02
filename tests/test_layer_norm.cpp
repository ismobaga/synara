#include <cassert>
#include <cmath>

#include "synara/nn/layer_norm.hpp"

int main()
{
    using namespace synara;

    LayerNorm ln(3, false);
    Tensor x = Tensor::from_vector(Shape({2, 3}), {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
    });

    Tensor y = ln(x);

    for (Size n = 0; n < 2; ++n)
    {
        double mean = 0.0;
        for (Size f = 0; f < 3; ++f)
        {
            mean += y.at({n, f});
        }
        mean /= 3.0;
        assert(std::fabs(mean) < 1e-5);

        double var = 0.0;
        for (Size f = 0; f < 3; ++f)
        {
            const double d = y.at({n, f}) - mean;
            var += d * d;
        }
        var /= 3.0;
        assert(std::fabs(var - 1.0) < 1e-4);
    }

    return 0;
}