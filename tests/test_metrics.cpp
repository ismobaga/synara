#include <cassert>
#include <cmath>

#include "synara/metrics/metrics.hpp"

int main()
{
    using namespace synara;

    Tensor logits = Tensor::from_vector(
        Shape({3, 2}),
        {
            0.1f,
            0.9f,
            0.6f,
            0.4f,
            0.7f,
            0.3f,
        },
        false);
    Tensor targets = Tensor::from_vector(Shape({3}), {1.0f, 0.0f, 1.0f}, false);
    const double acc = accuracy(logits, targets);
    assert(std::fabs(acc - (2.0 / 3.0)) < 1e-12);

    Tensor pred_bin = Tensor::from_vector(Shape({4}), {0.1f, 0.6f, 0.2f, 0.7f}, false);
    Tensor tgt_bin = Tensor::from_vector(Shape({4}), {0.0f, 1.0f, 1.0f, 1.0f}, false);
    const double bacc = binary_accuracy(pred_bin, tgt_bin, 0.5f);
    assert(std::fabs(bacc - 0.75) < 1e-12);

    return 0;
}
