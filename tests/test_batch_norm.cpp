#include <cassert>
#include <cmath>

#include "synara/nn/batch_norm.hpp"

int main()
{
    using namespace synara;

    BatchNorm1d bn(2, false);

    Tensor x = Tensor::from_vector(Shape({3, 2}), {
                                                     1.0f,
                                                     2.0f,
                                                     3.0f,
                                                     4.0f,
                                                     5.0f,
                                                     6.0f,
                                                 });

    Tensor y = bn(x);

    // Per-feature mean should be ~0 in training mode.
    const float mean0 = (y.at({0, 0}) + y.at({1, 0}) + y.at({2, 0})) / 3.0f;
    const float mean1 = (y.at({0, 1}) + y.at({1, 1}) + y.at({2, 1})) / 3.0f;

    assert(std::fabs(mean0) < 1e-5f);
    assert(std::fabs(mean1) < 1e-5f);

    // Running stats should have been updated after one training forward.
    assert(std::fabs(bn.running_mean().at({0, 0})) > 1e-6f);
    assert(std::fabs(bn.running_mean().at({0, 1})) > 1e-6f);

    bn.eval();
    Tensor y_eval = bn(x);
    assert(y_eval.shape() == x.shape());

    return 0;
}
