#include <cassert>
#include <cmath>

#include "synara/nn/batch_norm.hpp"

int main()
{
    using namespace synara;

    BatchNorm2d bn(2, false);

    Tensor x = Tensor::from_vector(
        Shape({1, 2, 2, 2}),
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
            10.0f, 20.0f,
            30.0f, 40.0f,
        });

    Tensor y = bn(x);

    float mean0 = 0.0f;
    float mean1 = 0.0f;
    for (size_t h = 0; h < 2; ++h)
    {
        for (size_t w = 0; w < 2; ++w)
        {
            mean0 += y.at({0, 0, h, w});
            mean1 += y.at({0, 1, h, w});
        }
    }
    mean0 /= 4.0f;
    mean1 /= 4.0f;

    assert(std::fabs(mean0) < 1e-5f);
    assert(std::fabs(mean1) < 1e-5f);

    assert(std::fabs(bn.running_mean().at({0, 0})) > 1e-6f);
    assert(std::fabs(bn.running_mean().at({0, 1})) > 1e-6f);

    bn.eval();
    Tensor y_eval = bn(x);
    assert(y_eval.shape() == x.shape());

    return 0;
}
