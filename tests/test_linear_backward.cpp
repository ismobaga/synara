#include <cassert>

#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;
    Linear layer(2, 1, true);
    auto x = Tensor::from_vector(Shape({2, 2}), {1.0f, 2.0f,
                                                 3.0f, 4.0f});

    auto target = Tensor::from_vector(Shape({2, 1}), {1.0f, 2.0f});

    auto pred = layer(x);
    auto loss = mse_loss(pred, target);
    loss.backward();
    auto parameters = layer.parameters();
    assert(parameters.size() == 2);
    auto &weight = parameters[0]->tensor();
    auto &bias = parameters[1]->tensor();
    assert(weight.shape() == Shape({1, 2}));
    assert(bias.shape() == Shape({1, 1}));
    assert(weight.grad().shape() == Shape({1, 2}));
    assert(bias.grad().shape() == Shape({1, 1}));

    return 0;
}
