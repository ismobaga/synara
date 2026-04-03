#include <cassert>
#include <cmath>

#include "synara/ops/activation.hpp"
#include "synara/ops/reduction.hpp"

namespace
{

    void test_gelu_forward_positive()
    {
        using namespace synara;

        // For large positive x, GELU(x) ≈ x
        auto a = Tensor::from_vector({1}, {5.0f});
        auto r = gelu(a);
        assert(std::abs(r.at({0}) - 5.0f) < 0.01f);
    }

    void test_gelu_forward_zero()
    {
        using namespace synara;

        // GELU(0) = 0
        auto a = Tensor::from_vector({1}, {0.0f});
        auto r = gelu(a);
        assert(std::abs(r.at({0})) < 1e-6f);
    }

    void test_gelu_forward_negative()
    {
        using namespace synara;

        // For large negative x, GELU(x) ≈ 0
        auto a = Tensor::from_vector({1}, {-5.0f});
        auto r = gelu(a);
        assert(std::abs(r.at({0})) < 0.01f);
    }

    void test_gelu_forward_batch()
    {
        using namespace synara;

        // GELU has a minimum near x≈-0.77, so it is NOT monotone on the full negative axis.
        // GELU(-2)≈-0.046 > GELU(-1)≈-0.159 > local-min, then increases for x > -0.77.
        auto a = Tensor::from_vector({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
        auto r = gelu(a);
        assert(r.at({0}) > r.at({1})); // GELU(-2) > GELU(-1): non-monotone region
        assert(r.at({1}) < r.at({2})); // GELU(-1) < GELU(1)
        assert(r.at({2}) < r.at({3})); // GELU(1) < GELU(2)
    }

    void test_gelu_no_grad()
    {
        using namespace synara;

        auto a = Tensor::from_vector({3}, {-1.0f, 0.0f, 1.0f});
        auto r = gelu(a);
        assert(!r.requires_grad());
    }

    void test_gelu_backward()
    {
        using namespace synara;

        auto x = Tensor::from_vector({3}, {-1.0f, 0.0f, 1.0f}, true);
        auto y = gelu(x);
        auto s = sum(y);
        s.backward();

        assert(x.has_grad());

        // Derivative at x=0: d/dx GELU(0) = 0.5
        assert(std::abs(x.grad().at({1}) - 0.5f) < 1e-4f);

        // By symmetry, d/dx GELU(-1) should be less than d/dx GELU(1)
        assert(x.grad().at({0}) < x.grad().at({2}));
    }

} // namespace

int main()
{
    test_gelu_forward_positive();
    test_gelu_forward_zero();
    test_gelu_forward_negative();
    test_gelu_forward_batch();
    test_gelu_no_grad();
    test_gelu_backward();
    return 0;
}
