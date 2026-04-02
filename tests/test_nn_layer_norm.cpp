#include <cassert>
#include <cmath>

#include "synara/nn/layer_norm.hpp"

int main()
{
    using namespace synara;

    LayerNorm ln(2, true);
    ln.weight().tensor().at({0, 0}) = 2.0;
    ln.weight().tensor().at({0, 1}) = 3.0;
    ln.bias().tensor().at({0, 0}) = -1.0;
    ln.bias().tensor().at({0, 1}) = 0.5;

    Tensor x = Tensor::from_vector(Shape({1, 2}), {1.0, 3.0});
    Tensor y = ln(x);

    const double inv_std = 1.0 / std::sqrt(1.0 + 1e-5);
    const double xhat0 = -inv_std;
    const double xhat1 = inv_std;

    assert(std::fabs(y.at({0, 0}) - (2.0 * xhat0 - 1.0)) < 1e-5);
    assert(std::fabs(y.at({0, 1}) - (3.0 * xhat1 + 0.5)) < 1e-5);
    assert(ln.parameters().size() == 2);

    return 0;
}