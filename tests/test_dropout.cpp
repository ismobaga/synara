#include <cassert>
#include <cmath>

#include "synara/nn/dropout.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Dropout d(0.5f, 12345ULL);

    Tensor x = Tensor::from_vector(Shape({2, 4}), {
                                                    1.0f,
                                                    2.0f,
                                                    3.0f,
                                                    4.0f,
                                                    5.0f,
                                                    6.0f,
                                                    7.0f,
                                                    8.0f,
                                                }, true);

    Tensor y = d(x);

    // Inverted dropout: non-zero outputs should be doubled when p=0.5.
    bool saw_zero = false;
    bool saw_scaled = false;
    for (Size i = 0; i < x.numel(); ++i)
    {
        if (std::fabs(y.data()[i]) < 1e-7f)
        {
            saw_zero = true;
        }
        else
        {
            saw_scaled = true;
            assert(std::fabs(y.data()[i] - (2.0f * x.data()[i])) < 1e-5f);
        }
    }
    assert(saw_zero);
    assert(saw_scaled);

    Tensor s = sum(y);
    s.backward();

    // Gradient must follow the same dropout mask and scaling.
    for (Size i = 0; i < x.numel(); ++i)
    {
        if (std::fabs(y.data()[i]) < 1e-7f)
        {
            assert(std::fabs(x.grad().data()[i]) < 1e-7f);
        }
        else
        {
            assert(std::fabs(x.grad().data()[i] - 2.0f) < 1e-5f);
        }
    }

    d.eval();
    Tensor y_eval = d(x);
    for (Size i = 0; i < x.numel(); ++i)
    {
        assert(std::fabs(y_eval.data()[i] - x.data()[i]) < 1e-7f);
    }

    return 0;
}
