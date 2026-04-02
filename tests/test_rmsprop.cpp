#include <cassert>
#include <cmath>
#include <memory>

#include "synara/nn/linear.hpp"
#include "synara/optim/rmsprop.hpp"
#include "synara/ops/loss.hpp"

namespace
{

void test_rmsprop_basic_step()
{
    using namespace synara;

    auto linear = std::make_shared<Linear>(2, 1, true);

    std::vector<Tensor *> tensors;
    for (Parameter *p : linear->parameters())
        tensors.push_back(&p->tensor());

    RMSprop optimizer(tensors, 0.01);

    for (int i = 0; i < 3; ++i)
    {
        optimizer.zero_grad();
        Tensor x   = Tensor::from_vector(Shape({1, 2}), {1.0f, 2.0f}, true);
        Tensor pred = linear->forward(x);
        Tensor loss = mse_loss(pred, Tensor::from_vector(Shape({1, 1}), {0.0f}, false));
        loss.backward();
        optimizer.step();
    }

    // If we reach here without throwing, the basic step works.
}

void test_rmsprop_converges()
{
    using namespace synara;

    // Minimise f(w) = (w - 3)^2 starting from w = 0
    Tensor w = Tensor::from_vector(Shape({}), {0.0f}, true);

    std::vector<Tensor *> params = {&w};
    RMSprop optimizer(params, RMSpropOptions{.lr = 0.1, .alpha = 0.9});

    for (int i = 0; i < 500; ++i)
    {
        optimizer.zero_grad();
        Tensor target = Tensor::from_vector(Shape({}), {3.0f}, false);
        Tensor loss   = mse_loss(w, target);
        loss.backward();
        optimizer.step();
    }

    // Should converge to w ≈ 3
    assert(std::abs(w.data()[0] - 3.0f) < 0.1f);
}

void test_rmsprop_options_validation()
{
    using namespace synara;

    Tensor w = Tensor::from_vector(Shape({}), {1.0f}, true);
    std::vector<Tensor *> params = {&w};

    bool caught = false;
    try
    {
        RMSprop bad(params, RMSpropOptions{.lr = -1.0});
    }
    catch (const ValueError &)
    {
        caught = true;
    }
    assert(caught);
}

void test_rmsprop_with_momentum()
{
    using namespace synara;

    Tensor w = Tensor::from_vector(Shape({}), {0.0f}, true);
    std::vector<Tensor *> params = {&w};
    RMSprop optimizer(params, RMSpropOptions{.lr = 0.05, .momentum = 0.9});

    for (int i = 0; i < 500; ++i)
    {
        optimizer.zero_grad();
        Tensor target = Tensor::from_vector(Shape({}), {3.0f}, false);
        Tensor loss   = mse_loss(w, target);
        loss.backward();
        optimizer.step();
    }

    assert(std::abs(w.data()[0] - 3.0f) < 0.1f);
}

} // namespace

int main()
{
    test_rmsprop_basic_step();
    test_rmsprop_converges();
    test_rmsprop_options_validation();
    test_rmsprop_with_momentum();
    return 0;
}
