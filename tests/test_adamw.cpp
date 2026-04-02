#include <cassert>
#include <cmath>
#include <memory>

#include "synara/core/error.hpp"
#include "synara/nn/linear.hpp"
#include "synara/optim/adamw.hpp"
#include "synara/ops/loss.hpp"

namespace
{

void test_adamw_basic_step()
{
    using namespace synara;

    auto linear = std::make_shared<Linear>(2, 1, true);

    std::vector<Tensor *> tensors;
    for (Parameter *p : linear->parameters())
        tensors.push_back(&p->tensor());

    AdamW optimizer(tensors, 0.001);

    for (int i = 0; i < 3; ++i)
    {
        optimizer.zero_grad();
        Tensor x    = Tensor::from_vector(Shape({1, 2}), {1.0f, 2.0f}, true);
        Tensor pred = linear->forward(x);
        Tensor loss = mse_loss(pred, Tensor::from_vector(Shape({1, 1}), {0.0f}));
        loss.backward();
        optimizer.step();
    }
    // Reaching here without exception means step() works.
}

void test_adamw_converges()
{
    using namespace synara;

    // Minimise f(w) = (w - 2)^2 starting from w = 0, with weight decay
    Tensor w = Tensor::from_vector(Shape({}), {0.0f}, true);

    std::vector<Tensor *> params = {&w};
    AdamW optimizer(params, AdamWOptions{.lr = 0.05, .weight_decay = 0.0});

    for (int i = 0; i < 1000; ++i)
    {
        optimizer.zero_grad();
        Tensor target = Tensor::from_vector(Shape({}), {2.0f});
        Tensor loss   = mse_loss(w, target);
        loss.backward();
        optimizer.step();
    }

    assert(std::abs(w.data()[0] - 2.0f) < 0.1f);
}

void test_adamw_weight_decay_shrinks_params()
{
    using namespace synara;

    // With large weight decay and zero gradients (target == current value),
    // the parameter should trend towards zero over many steps.
    Tensor w = Tensor::from_vector(Shape({}), {5.0f}, true);
    std::vector<Tensor *> params = {&w};
    // Set target equal to current value so the gradient signal is zero;
    // only weight decay should drive the parameter down.
    AdamW optimizer(params, AdamWOptions{.lr = 0.1, .weight_decay = 0.5});

    float initial = w.data()[0];
    for (int i = 0; i < 50; ++i)
    {
        optimizer.zero_grad();
        // Create a "zero gradient" scenario: predict w, target = w
        float cur = w.data()[0];
        Tensor target = Tensor::from_vector(Shape({}), {cur});
        Tensor loss   = mse_loss(w, target);
        loss.backward();
        optimizer.step();
    }

    // Parameter should have moved closer to 0 due to decoupled weight decay
    assert(std::abs(w.data()[0]) < std::abs(initial));
}

void test_adamw_options_validation()
{
    using namespace synara;

    Tensor w = Tensor::from_vector(Shape({}), {1.0f}, true);
    std::vector<Tensor *> params = {&w};

    bool caught = false;
    try { AdamW bad(params, AdamWOptions{.lr = -0.001}); }
    catch (const ValueError &) { caught = true; }
    assert(caught);

    caught = false;
    try { AdamW bad(params, AdamWOptions{.beta1 = 1.0}); }
    catch (const ValueError &) { caught = true; }
    assert(caught);

    caught = false;
    try { AdamW bad(params, AdamWOptions{.weight_decay = -0.1}); }
    catch (const ValueError &) { caught = true; }
    assert(caught);
}

} // namespace

int main()
{
    test_adamw_basic_step();
    test_adamw_converges();
    test_adamw_weight_decay_shrinks_params();
    test_adamw_options_validation();
    return 0;
}
