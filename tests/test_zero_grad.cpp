#include <cassert>

#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"

int main()
{
    using namespace synara;

    Linear layer(2, 1, true);
    Tensor x = Tensor::from_vector(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target = Tensor::from_vector(Shape({2, 1}), {1.0f, 2.0f});

    Tensor pred = layer(x);
    Tensor loss = mse_loss(pred, target);
    loss.backward();

    auto params = layer.parameters();
    assert(params.size() == 2);

    Tensor &w = params[0]->tensor();
    Tensor &b = params[1]->tensor();

    assert(w.has_grad());
    assert(b.has_grad());

    bool has_non_zero_grad = false;
    for (std::size_t i = 0; i < w.numel(); ++i)
    {
        if (w.grad().data()[i] != 0.0f)
        {
            has_non_zero_grad = true;
            break;
        }
    }
    for (std::size_t i = 0; i < b.numel(); ++i)
    {
        if (b.grad().data()[i] != 0.0f)
        {
            has_non_zero_grad = true;
            break;
        }
    }
    assert(has_non_zero_grad);

    layer.zero_grad();

    for (std::size_t i = 0; i < w.numel(); ++i)
    {
        assert(w.grad().data()[i] == 0.0f);
    }
    for (std::size_t i = 0; i < b.numel(); ++i)
    {
        assert(b.grad().data()[i] == 0.0f);
    }

    return 0;
}
