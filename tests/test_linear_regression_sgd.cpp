#include <cassert>
#include <cmath>
#include <vector>

#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"
#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    Linear model(1, 1, true);

    Tensor x = Tensor::from_vector(Shape({4, 1}), {-1.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = Tensor::from_vector(Shape({4, 1}), {-1.0f, 1.0f, 3.0f, 5.0f});

    std::vector<Parameter *> params = model.parameters();
    std::vector<Tensor *> tensors;
    tensors.reserve(params.size());
    for (Parameter *p : params)
    {
        tensors.push_back(&p->tensor());
    }

    SGD optim(tensors, 0.05);

    Tensor::value_type initial_loss = mse_loss(model(x), y).item();

    for (int step = 0; step < 250; ++step)
    {
        optim.zero_grad();
        Tensor pred = model(x);
        Tensor loss = mse_loss(pred, y);
        loss.backward();
        optim.step();
    }

    Tensor::value_type final_loss = mse_loss(model(x), y).item();

    assert(final_loss < initial_loss);
    assert(final_loss < 1e-2f);

    Tensor::value_type learned_w = model.weight().tensor().at({0, 0});
    Tensor::value_type learned_b = model.bias().tensor().at({0, 0});

    assert(std::fabs(learned_w - 2.0f) < 0.15f);
    assert(std::fabs(learned_b - 1.0f) < 0.15f);

    return 0;
}
