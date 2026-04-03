#pragma once

#include <unordered_map>

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"
namespace synara
{

    struct SGDOptions
    {
        double lr = 0.01;
        double momentum = 0.0;
        double weight_decay = 0.0;
        double max_grad_norm = 0.0;
    };

    class SGD : public Optimizer
    {
    public:
        SGD(std::vector<Tensor *> params, double lr = 0.01);
        SGD(std::vector<Tensor *> params, SGDOptions options);

        void step() override;

        double learning_rate() const noexcept override { return options_.lr; }
        void set_learning_rate(double lr) noexcept override { options_.lr = lr; }

        double momentum() const noexcept { return options_.momentum; }
        void set_momentum(double momentum) noexcept { options_.momentum = momentum; }

        double weight_decay() const noexcept { return options_.weight_decay; }
        void set_weight_decay(double weight_decay) noexcept { options_.weight_decay = weight_decay; }

        double max_grad_norm() const noexcept { return options_.max_grad_norm; }
        void set_max_grad_norm(double max_grad_norm) noexcept { options_.max_grad_norm = max_grad_norm; }

    private:
        SGDOptions options_;
        std::unordered_map<Tensor *, Tensor> velocity_;
    };

} // namespace synara