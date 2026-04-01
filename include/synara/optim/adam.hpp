#pragma once

#include <unordered_map>

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"

namespace synara
{

    struct AdamOptions
    {
        double lr = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double weight_decay = 0.0;
    };

    class Adam : public Optimizer
    {
    public:
        Adam(std::vector<Tensor *> params, double lr = 0.001);
        Adam(std::vector<Tensor *> params, AdamOptions options);

        void step() override;

        double learning_rate() const noexcept { return options_.lr; }
        void set_learning_rate(double lr) noexcept { options_.lr = lr; }

        double beta1() const noexcept { return options_.beta1; }
        void set_beta1(double beta1) noexcept { options_.beta1 = beta1; }

        double beta2() const noexcept { return options_.beta2; }
        void set_beta2(double beta2) noexcept { options_.beta2 = beta2; }

        double eps() const noexcept { return options_.eps; }
        void set_eps(double eps) noexcept { options_.eps = eps; }

        double weight_decay() const noexcept { return options_.weight_decay; }
        void set_weight_decay(double weight_decay) noexcept { options_.weight_decay = weight_decay; }

    private:
        AdamOptions options_;
        std::unordered_map<Tensor *, Tensor> m_;
        std::unordered_map<Tensor *, Tensor> v_;
        int step_count_ = 0;
    };

} // namespace synara
