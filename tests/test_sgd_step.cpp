#include <cassert>
#include <cmath>

#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    Tensor param = Tensor::from_vector(Shape({2}), {2.0f, -3.0f}, true);
    param.set_grad(Tensor::from_vector(Shape({2}), {0.5f, -1.0f}));

    SGD optim({&param}, 0.1);
    optim.step();

    assert(std::fabs(param.at({0}) - 1.95f) < 1e-6f);
    assert(std::fabs(param.at({1}) + 2.9f) < 1e-6f);

    optim.zero_grad();
    assert(param.grad().at({0}) == 0.0f);
    assert(param.grad().at({1}) == 0.0f);

    return 0;
}
