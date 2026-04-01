#include <cassert>
#include <memory>

#include "synara/nn/linear.hpp"
#include "synara/optim/adam.hpp"

int main()
{
    using namespace synara;

    std::vector<Tensor *> params;
    Adam optimizer(params, 0.001);

    assert(optimizer.learning_rate() == 0.001);
    optimizer.set_learning_rate(0.0001);
    assert(optimizer.learning_rate() == 0.0001);

    assert(optimizer.beta1() == 0.9);
    optimizer.set_beta1(0.95);
    assert(optimizer.beta1() == 0.95);

    assert(optimizer.beta2() == 0.999);
    optimizer.set_beta2(0.99);
    assert(optimizer.beta2() == 0.99);

    assert(optimizer.weight_decay() == 0.0);
    optimizer.set_weight_decay(0.0001);
    assert(optimizer.weight_decay() == 0.0001);

    return 0;
}
