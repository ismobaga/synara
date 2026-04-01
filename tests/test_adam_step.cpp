#include <cassert>
#include <memory>

#include "synara/nn/linear.hpp"
#include "synara/optim/adam.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    auto linear = std::make_shared<Linear>(2, 1, true);

    std::vector<Parameter *> params = linear->parameters();
    std::vector<Tensor *> tensors;
    for (Parameter *p : params)
    {
        tensors.push_back(&p->tensor());
    }

    Adam optimizer(tensors, 0.01);

    for (int i = 0; i < 3; ++i)
    {
        optimizer.zero_grad();

        Tensor x = Tensor::from_vector(Shape({1, 2}), {1.0f, 2.0f}, true);
        Tensor pred = linear->forward(x);
        Tensor loss = mse_loss(pred, Tensor::from_vector(Shape({1, 1}), {0.0f}, false));
        loss.backward();

        optimizer.step();
    }

    return 0;
}
