#include <cassert>
#include <cmath>
#include <vector>

#include "synara/ops/linalg.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    // ---------------------------------------------------------------
    // 1. Analytical gradients via backward()
    // ---------------------------------------------------------------
    auto a = Tensor::from_vector(Shape({2, 3}),
                                 {1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f},
                                 true);

    auto b = Tensor::from_vector(Shape({3, 2}),
                                 {7.0f,  8.0f,
                                  9.0f,  10.0f,
                                  11.0f, 12.0f},
                                 true);

    auto c = matmul(a, b);
    auto s = sum(c);
    s.backward();

    assert(a.has_grad());
    assert(b.has_grad());
    assert(a.grad().shape() == a.shape());
    assert(b.grad().shape() == b.shape());

    // For s = sum(matmul(a, b)):
    //   ds/da = ones(2,2) * b^T  =>  each row of grad_a = sum of rows of b^T = row sums of b
    //   ds/db = a^T * ones(2,2)  =>  each column of grad_b = sum of cols of a^T = col sums of a
    // b row sums: [7+8, 9+10, 11+12] = [15, 19, 23]
    assert(a.grad().at({0, 0}) == 15.0f);
    assert(a.grad().at({0, 1}) == 19.0f);
    assert(a.grad().at({0, 2}) == 23.0f);
    assert(a.grad().at({1, 0}) == 15.0f);
    assert(a.grad().at({1, 1}) == 19.0f);
    assert(a.grad().at({1, 2}) == 23.0f);

    // a col sums: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert(b.grad().at({0, 0}) == 5.0f);
    assert(b.grad().at({0, 1}) == 5.0f);
    assert(b.grad().at({1, 0}) == 7.0f);
    assert(b.grad().at({1, 1}) == 7.0f);
    assert(b.grad().at({2, 0}) == 9.0f);
    assert(b.grad().at({2, 1}) == 9.0f);

    // ---------------------------------------------------------------
    // 2. Finite-difference gradient check
    // ---------------------------------------------------------------
    auto a2 = Tensor::from_vector(Shape({2, 3}),
                                  {0.5f, -1.0f, 2.0f,
                                   -0.5f, 1.5f, -2.0f},
                                  true);

    auto b2 = Tensor::from_vector(Shape({3, 2}),
                                  {1.0f, -1.0f,
                                   2.0f, 0.5f,
                                   -1.5f, 3.0f},
                                  true);

    auto c2 = matmul(a2, b2);
    auto s2 = sum(c2);
    s2.backward();

    std::vector<float> a2_grad(a2.grad().data(),
                               a2.grad().data() + a2.grad().numel());
    std::vector<float> b2_grad(b2.grad().data(),
                               b2.grad().data() + b2.grad().numel());

    const float eps     = 1e-4f;
    const float abs_tol = 1e-3f;
    const float rel_tol = 1e-2f;

    auto loss_fn = [&]() {
        return sum(matmul(a2, b2)).item();
    };

    // Check a2 gradients
    for (std::size_t i = 0; i < a2.numel(); ++i)
    {
        float orig   = a2.data()[i];
        a2.data()[i] = orig + eps;
        float fp     = loss_fn();
        a2.data()[i] = orig - eps;
        float fm     = loss_fn();
        a2.data()[i] = orig;

        float fd  = (fp - fm) / (2.0f * eps);
        float ana = a2_grad[i];

        float abs_err = std::fabs(fd - ana);
        float rel_err = abs_err / (std::fabs(ana) + 1e-8f);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    // Check b2 gradients
    for (std::size_t i = 0; i < b2.numel(); ++i)
    {
        float orig   = b2.data()[i];
        b2.data()[i] = orig + eps;
        float fp     = loss_fn();
        b2.data()[i] = orig - eps;
        float fm     = loss_fn();
        b2.data()[i] = orig;

        float fd  = (fp - fm) / (2.0f * eps);
        float ana = b2_grad[i];

        float abs_err = std::fabs(fd - ana);
        float rel_err = abs_err / (std::fabs(ana) + 1e-8f);
        assert(abs_err < abs_tol || rel_err < rel_tol);
    }

    return 0;
}
