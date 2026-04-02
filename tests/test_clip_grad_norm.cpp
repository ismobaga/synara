#include <cassert>
#include <cmath>
#include <stdexcept>

#include "synara/optim/clip_grad.hpp"
#include "synara/tensor/tensor.hpp"

namespace
{

void test_clip_grad_norm_no_clip()
{
    using namespace synara;

    // Gradient norm is below max_norm => no clipping, returned norm is correct
    Tensor w = Tensor::from_vector(Shape({3}), {1.0f, 1.0f, 1.0f}, true);
    w.set_grad(Tensor::from_vector(Shape({3}), {1.0f, 0.0f, 0.0f}));

    float norm = clip_grad_norm({&w}, 10.0f);

    assert(std::fabs(norm - 1.0f) < 1e-5f);
    // Gradient should be unchanged
    assert(w.grad().data()[0] == 1.0f);
    assert(w.grad().data()[1] == 0.0f);
    assert(w.grad().data()[2] == 0.0f);
}

void test_clip_grad_norm_clips()
{
    using namespace synara;

    // Gradient: [3, 4] => norm = 5; max_norm = 1 => scale = 0.2
    Tensor w = Tensor::from_vector(Shape({2}), {0.0f, 0.0f}, true);
    w.set_grad(Tensor::from_vector(Shape({2}), {3.0f, 4.0f}));

    float norm = clip_grad_norm({&w}, 1.0f);

    assert(std::fabs(norm - 5.0f) < 1e-4f);
    assert(std::fabs(w.grad().data()[0] - 0.6f) < 1e-5f);
    assert(std::fabs(w.grad().data()[1] - 0.8f) < 1e-5f);
}

void test_clip_grad_norm_multiple_params()
{
    using namespace synara;

    // Two params: grad norms [3,4] and [0,5] => total = sqrt(9+16+0+25) = sqrt(50)
    Tensor w1 = Tensor::from_vector(Shape({2}), {0.0f, 0.0f}, true);
    w1.set_grad(Tensor::from_vector(Shape({2}), {3.0f, 4.0f}));

    Tensor w2 = Tensor::from_vector(Shape({2}), {0.0f, 0.0f}, true);
    w2.set_grad(Tensor::from_vector(Shape({2}), {0.0f, 5.0f}));

    const float expected_norm = std::sqrt(50.0f);
    const float max_norm      = 1.0f;
    float norm = clip_grad_norm({&w1, &w2}, max_norm);

    assert(std::fabs(norm - expected_norm) < 1e-4f);

    // After clipping the norm of the combined gradient vector should be max_norm
    double sq = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        sq += w1.grad().data()[i] * w1.grad().data()[i];
        sq += w2.grad().data()[i] * w2.grad().data()[i];
    }
    assert(std::fabs(static_cast<float>(std::sqrt(sq)) - max_norm) < 1e-5f);
}

void test_clip_grad_norm_invalid_max_norm()
{
    using namespace synara;

    Tensor w = Tensor::from_vector(Shape({1}), {1.0f}, true);
    w.set_grad(Tensor::from_vector(Shape({1}), {1.0f}));

    bool caught = false;
    try { clip_grad_norm({&w}, -1.0f); }
    catch (const std::invalid_argument &) { caught = true; }
    assert(caught);
}

void test_clip_grad_norm_skips_no_grad()
{
    using namespace synara;

    // Tensor without requires_grad should be ignored
    Tensor w = Tensor::from_vector(Shape({2}), {0.0f, 0.0f}, false);

    float norm = clip_grad_norm({&w}, 1.0f);
    assert(norm == 0.0f);
}

} // namespace

int main()
{
    test_clip_grad_norm_no_clip();
    test_clip_grad_norm_clips();
    test_clip_grad_norm_multiple_params();
    test_clip_grad_norm_invalid_max_norm();
    test_clip_grad_norm_skips_no_grad();
    return 0;
}
