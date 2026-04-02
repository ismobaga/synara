#include <cassert>
#include <cmath>

#include "synara/ops/linalg.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    // A: (2, 3), B: (3, 2) => C = A*B: (2, 2)
    auto a = Tensor::from_vector(Shape({2, 3}),
                                 {1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f},
                                 true);

    auto b = Tensor::from_vector(Shape({3, 2}),
                                 {7.0f,  8.0f,
                                  9.0f,  10.0f,
                                  11.0f, 12.0f},
                                 true);

    // Forward + backward
    auto c = matmul(a, b);
    auto s = sum(c);
    s.backward();

    assert(a.has_grad());
    assert(b.has_grad());
    assert(a.grad().shape() == a.shape());
    assert(b.grad().shape() == b.shape());

    // dL/dA = dL/dC * B^T  (dL/dC is all-ones because loss = sum(C))
    // Row-sum of B: [7+8, 9+10, 11+12] = [15, 19, 23] for every row of grad_A
    assert(a.grad().at({0, 0}) == 15.0f);
    assert(a.grad().at({0, 1}) == 19.0f);
    assert(a.grad().at({0, 2}) == 23.0f);
    assert(a.grad().at({1, 0}) == 15.0f);
    assert(a.grad().at({1, 1}) == 19.0f);
    assert(a.grad().at({1, 2}) == 23.0f);

    // dL/dB = A^T * dL/dC  (dL/dC is all-ones)
    // Col-sum of A: [1+4, 2+5, 3+6] = [5, 7, 9] for every column of grad_B
    assert(b.grad().at({0, 0}) == 5.0f);
    assert(b.grad().at({0, 1}) == 5.0f);
    assert(b.grad().at({1, 0}) == 7.0f);
    assert(b.grad().at({1, 1}) == 7.0f);
    assert(b.grad().at({2, 0}) == 9.0f);
    assert(b.grad().at({2, 1}) == 9.0f);

    // Also validate against finite differences
    const float eps     = 1e-3f;
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-2f;

    std::vector<float> grad_a(a.grad().data(), a.grad().data() + a.numel());
    std::vector<float> grad_b(b.grad().data(), b.grad().data() + b.numel());

    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        float orig    = a.data()[i];
        a.data()[i]   = orig + eps;
        float f_plus  = sum(matmul(a, b)).item();
        a.data()[i]   = orig - eps;
        float f_minus = sum(matmul(a, b)).item();
        a.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_a[i]);
        assert(err < abs_tol || err / (std::fabs(grad_a[i]) + 1e-8f) < rel_tol);
    }

    for (std::size_t i = 0; i < b.numel(); ++i)
    {
        float orig    = b.data()[i];
        b.data()[i]   = orig + eps;
        float f_plus  = sum(matmul(a, b)).item();
        b.data()[i]   = orig - eps;
        float f_minus = sum(matmul(a, b)).item();
        b.data()[i]   = orig;

        float fd  = (f_plus - f_minus) / (2.0f * eps);
        float err = std::fabs(fd - grad_b[i]);
        assert(err < abs_tol || err / (std::fabs(grad_b[i]) + 1e-8f) < rel_tol);
    }

    return 0;
}
