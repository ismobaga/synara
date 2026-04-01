#include <cassert>
#include <cmath>

#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    // Momentum + weight decay deterministic check across two steps.
    Tensor p = Tensor::from_vector(Shape({2}), {1.0f, -2.0f}, true);
    SGDOptions opts;
    opts.lr = 0.1;
    opts.momentum = 0.9;
    opts.weight_decay = 0.1;
    SGD optim({&p}, opts);

    p.set_grad(Tensor::from_vector(Shape({2}), {0.5f, -1.0f}));
    optim.step();
    assert(std::fabs(p.at({0}) - 0.94f) < 1e-6f);
    assert(std::fabs(p.at({1}) + 1.88f) < 1e-6f);

    p.set_grad(Tensor::from_vector(Shape({2}), {0.5f, -1.0f}));
    optim.step();
    assert(std::fabs(p.at({0}) - 0.8266f) < 1e-4f);
    assert(std::fabs(p.at({1}) + 1.6532f) < 1e-4f);

    // Gradient clipping by norm check.
    Tensor c = Tensor::from_vector(Shape({1}), {0.0f}, true);
    SGDOptions clip_opts;
    clip_opts.lr = 1.0;
    clip_opts.max_grad_norm = 1.0;
    SGD clip_optim({&c}, clip_opts);

    c.set_grad(Tensor::from_vector(Shape({1}), {10.0f}));
    clip_optim.step();
    assert(std::fabs(c.at({0}) + 1.0f) < 1e-6f);

    return 0;
}
