#include <iostream>
#include <memory>
#include <vector>

#include "synara/nn/linear.hpp"
#include "synara/nn/relu.hpp"
#include "synara/nn/sequential.hpp"
#include "synara/nn/sigmoid.hpp"
#include "synara/ops/loss.hpp"
#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    auto l1 = std::make_shared<Linear>(2, 8, true);
    auto a1 = std::make_shared<ReLU>();
    auto l2 = std::make_shared<Linear>(8, 1, true);
    auto a2 = std::make_shared<Sigmoid>();

    Sequential model({l1, a1, l2, a2});

    Tensor x = Tensor::from_vector(
        Shape({4, 2}),
        {
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
        });

    Tensor y = Tensor::from_vector(Shape({4, 1}), {0.0f, 1.0f, 1.0f, 0.0f});

    std::vector<Parameter *> params = model.parameters();
    std::vector<Tensor *> tensors;
    tensors.reserve(params.size());
    for (Parameter *p : params)
    {
        tensors.push_back(&p->tensor());
    }

    SGD optimizer(tensors, 0.05);

    for (int epoch = 0; epoch < 5000; ++epoch)
    {
        optimizer.zero_grad();

        Tensor pred = model(x);
        Tensor loss = binary_cross_entropy(pred, y);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 500 == 0)
        {
            std::cout << "epoch " << (epoch + 1) << " loss=" << loss.item() << "\n";
        }
    }

    Tensor pred = model(x);
    std::cout << "\nXOR predictions (probabilities):\n";
    for (std::size_t i = 0; i < 4; ++i)
    {
        const float prob = pred.at({i, 0});
        const int cls = (prob >= 0.5f) ? 1 : 0;
        std::cout << "x=[" << x.at({i, 0}) << ", " << x.at({i, 1}) << "]"
                  << " p1=" << prob
                  << " class=" << cls
                  << " target=" << y.at({i, 0}) << "\n";
    }

    std::cout << "\nFinal parameters:\n";
    std::cout << "l1 weight: " << l1->weight().tensor() << "\n";
    std::cout << "l1 bias: " << l1->bias().tensor() << "\n";
    std::cout << "l2 weight: " << l2->weight().tensor() << "\n";
    std::cout << "l2 bias: " << l2->bias().tensor() << "\n";

    return 0;
}
