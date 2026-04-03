#include <cassert>
#include <cmath>

#include "synara/optim/grad_clip.hpp"

int main()
{
    using namespace synara;

    Tensor p(Shape({2}), true);
    Tensor g = Tensor::from_vector(Shape({2}), {3.0f, 4.0f}, false);
    p.set_grad(g);

    std::vector<Tensor *> params = {&p};
    const double norm = clip_grad_norm_(params, 1.0);
    assert(std::fabs(norm - 5.0) < 1e-12);

    const double new_norm = std::sqrt(
        static_cast<double>(p.grad().data()[0]) * p.grad().data()[0] +
        static_cast<double>(p.grad().data()[1]) * p.grad().data()[1]);
    assert(std::fabs(new_norm - 1.0) < 1e-6);

    clip_grad_value_(params, 0.2);
    assert(std::fabs(p.grad().data()[0]) <= 0.2f + 1e-6f);
    assert(std::fabs(p.grad().data()[1]) <= 0.2f + 1e-6f);

    return 0;
}
