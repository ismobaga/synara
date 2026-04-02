#pragma once

#include <unordered_map>

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"

namespace synara
{

    struct RMSpropOptions
    {
        double lr          = 0.01;
        double alpha       = 0.99;  ///< smoothing constant
        double eps         = 1e-8;
        double weight_decay = 0.0;
        double momentum    = 0.0;
    };

    /// RMSprop optimiser.
    ///
    /// Update rule (no momentum):
    ///   v  = alpha * v + (1 - alpha) * g^2
    ///   p -= lr * g / (sqrt(v) + eps)
    ///
    /// With momentum:
    ///   v  = alpha * v + (1 - alpha) * g^2
    ///   buf = momentum * buf + g / (sqrt(v) + eps)
    ///   p -= lr * buf
    class RMSprop : public Optimizer
    {
    public:
        RMSprop(std::vector<Tensor *> params, double lr = 0.01);
        RMSprop(std::vector<Tensor *> params, RMSpropOptions options);

        void step() override;

        double learning_rate() const noexcept { return options_.lr; }
        void set_learning_rate(double lr) noexcept { options_.lr = lr; }

        double alpha() const noexcept { return options_.alpha; }
        void set_alpha(double alpha) noexcept { options_.alpha = alpha; }

        double eps() const noexcept { return options_.eps; }
        void set_eps(double eps) noexcept { options_.eps = eps; }

        double weight_decay() const noexcept { return options_.weight_decay; }
        void set_weight_decay(double wd) noexcept { options_.weight_decay = wd; }

        double momentum() const noexcept { return options_.momentum; }
        void set_momentum(double m) noexcept { options_.momentum = m; }

    private:
        RMSpropOptions options_;
        std::unordered_map<Tensor *, Tensor> square_avg_;
        std::unordered_map<Tensor *, Tensor> momentum_buf_;
    };

} // namespace synara
