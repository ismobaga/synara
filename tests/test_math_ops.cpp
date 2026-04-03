#include <cassert>
#include <cmath>

#include "synara/ops/math.hpp"
#include "synara/ops/reduction.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(Shape({2}), {1.0f, 4.0f}, true);
    Tensor y = sqrt(x);
    assert(std::fabs(y.data()[0] - 1.0f) < 1e-6);
    assert(std::fabs(y.data()[1] - 2.0f) < 1e-6);

    Tensor z = log(exp(x));
    assert(std::fabs(z.data()[0] - 1.0f) < 1e-5);
    assert(std::fabs(z.data()[1] - 4.0f) < 1e-5);

    Tensor c = clamp(x, 0.5f, 2.0f);
    assert(c.data()[0] == 1.0f);
    assert(c.data()[1] == 2.0f);

    Tensor loss = sum(y);
    loss.backward();
    assert(x.has_grad());
    assert(std::fabs(x.grad().data()[0] - 0.5f) < 1e-5);
    assert(std::fabs(x.grad().data()[1] - 0.25f) < 1e-5);

    return 0;
}
