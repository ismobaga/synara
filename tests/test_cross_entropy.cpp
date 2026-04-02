#include <cassert>
#include <cmath>

#include "synara/ops/loss.hpp"
#include "synara/ops/reduction.hpp"

namespace
{

void test_log_softmax_sums_to_zero()
{
    using namespace synara;

    // log(softmax) values for a row must sum to log(1) == 0 after exp.
    // Equivalently, sum(exp(log_softmax)) == 1.
    auto a   = Tensor::from_vector({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto lsm = log_softmax(a, 1);

    float sum_exp = 0.0f;
    for (int c = 0; c < 4; ++c)
        sum_exp += std::exp(lsm.at({0, static_cast<std::size_t>(c)}));
    assert(std::abs(sum_exp - 1.0f) < 1e-5f);
}

void test_log_softmax_two_rows()
{
    using namespace synara;

    auto a   = Tensor::from_vector({2, 3}, {1.0f, 2.0f, 3.0f,
                                            0.5f, 0.5f, 0.5f});
    auto lsm = log_softmax(a, 1);

    // Each row must independently satisfy sum(exp) == 1
    for (int n = 0; n < 2; ++n)
    {
        float s = 0.0f;
        for (int c = 0; c < 3; ++c)
            s += std::exp(lsm.at({static_cast<std::size_t>(n), static_cast<std::size_t>(c)}));
        assert(std::abs(s - 1.0f) < 1e-5f);
    }
}

void test_cross_entropy_uniform_target()
{
    using namespace synara;

    // With uniform logits the loss equals log(C).
    const int C = 4;
    auto logits  = Tensor::from_vector({1, C}, {0.0f, 0.0f, 0.0f, 0.0f}, true);
    auto targets = Tensor::from_vector({1, C}, {0.25f, 0.25f, 0.25f, 0.25f});

    auto loss = cross_entropy_loss(logits, targets);
    // Expected: -sum(0.25 * log(0.25)) = log(4)
    const float expected = std::log(static_cast<float>(C));
    assert(std::abs(loss.item() - expected) < 1e-5f);
}

void test_cross_entropy_one_hot()
{
    using namespace synara;

    // Perfect prediction: logits strongly favour class 0 => loss near 0
    auto logits  = Tensor::from_vector({1, 3}, {10.0f, 0.0f, 0.0f}, true);
    auto targets = Tensor::from_vector({1, 3}, {1.0f, 0.0f, 0.0f});

    auto loss = cross_entropy_loss(logits, targets);
    assert(loss.item() < 0.001f);
}

void test_cross_entropy_backward()
{
    using namespace synara;

    auto logits  = Tensor::from_vector({2, 3},
                                        {1.0f, 2.0f, 3.0f,
                                         1.0f, 1.0f, 1.0f}, true);
    auto targets = Tensor::from_vector({2, 3},
                                        {0.0f, 0.0f, 1.0f,
                                         0.333f, 0.333f, 0.334f});

    auto loss = cross_entropy_loss(logits, targets);
    loss.backward();

    assert(logits.has_grad());
    assert(logits.grad().shape() == logits.shape());

    // Gradient sum over classes for each row should be ~0 (softmax - target sums to 0)
    for (int n = 0; n < 2; ++n)
    {
        float row_sum = 0.0f;
        for (int c = 0; c < 3; ++c)
            row_sum += logits.grad().at({static_cast<std::size_t>(n), static_cast<std::size_t>(c)});
        assert(std::abs(row_sum) < 1e-4f);
    }
}

} // namespace

int main()
{
    test_log_softmax_sums_to_zero();
    test_log_softmax_two_rows();
    test_cross_entropy_uniform_target();
    test_cross_entropy_one_hot();
    test_cross_entropy_backward();
    return 0;
}
