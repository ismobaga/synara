#include <iostream>
#include <vector>

#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"
#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    Linear model(1, 1, true);

    Tensor x = Tensor::from_vector(Shape({4, 1}), {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::from_vector(Shape({4, 1}), {3.0f, 5.0f, 7.0f, 9.0f});

    std::vector<Parameter *> params = model.parameters();
    std::vector<Tensor *> param_tensors;
    param_tensors.reserve(params.size());
    for (Parameter *p : params)
    {
        param_tensors.push_back(&p->tensor());
    }

    SGD optimizer(param_tensors, 0.05);

    for (int epoch = 0; epoch < 300; ++epoch)
    {
        optimizer.zero_grad();
        Tensor pred = model(x);
        Tensor loss = mse_loss(pred, y);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 50 == 0)
        {
            std::cout << "epoch " << (epoch + 1)
                      << " loss=" << loss.item() << "\n";
        }
    }

    const float w = model.weight().tensor().at({0, 0});
    const float b = model.bias().tensor().at({0, 0});

    std::cout << "Learned weight: " << w << "\n";
    std::cout << "Learned bias: " << b << "\n";

    std::cout << "Predictions:" << "\n";
    Tensor pred = model(x);
    for (std::size_t i = 0; i < 4; ++i)
    {
        std::cout << "x=" << x.at({i, 0})
                  << " y_hat=" << pred.at({i, 0})
                  << " y=" << y.at({i, 0}) << "\n";
    }

    return 0;
}
