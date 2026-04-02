#pragma once

#include <unordered_map>

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"

namespace synara
{

    struct AdamWOptions
    {
        double lr           = 0.001;
        double beta1        = 0.9;
        double beta2        = 0.999;
        double eps          = 1e-8;
        double weight_decay = 0.01;  ///< decoupled weight decay (applied directly to params)
    };

    /// AdamW optimiser (Adam with decoupled weight decay).
    ///
    /// Unlike Adam, the weight-decay penalty is applied directly to the
    /// parameters rather than being folded into the gradient, which avoids
    /// the interaction between momentum estimates and regularisation:
    ///
    ///   p = p * (1 - lr * weight_decay)
    ///   m = beta1 * m + (1 - beta1) * g
    ///   v = beta2 * v + (1 - beta2) * g^2
    ///   p -= lr * m_hat / (sqrt(v_hat) + eps)
    ///
    /// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
    ///            (https://arxiv.org/abs/1711.05101)
    class AdamW : public Optimizer
    {
    public:
        AdamW(std::vector<Tensor *> params, double lr = 0.001);
        AdamW(std::vector<Tensor *> params, AdamWOptions options);

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
        AdamWOptions options_;
        std::unordered_map<Tensor *, Tensor> m_;
        std::unordered_map<Tensor *, Tensor> v_;
        int step_count_ = 0;
    };

} // namespace synara
