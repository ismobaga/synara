#pragma once

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"
namespace synara
{

    class SGD : public Optimizer
    {
    public:
        SGD(std::vector<Tensor *> params, double lr = 0.01);

        void step() override;

        double learning_rate() const noexcept { return lr_; }
        void set_learning_rate(double lr) noexcept { lr_ = lr; }

    private:
        double lr_;
    };

} // namespace synara