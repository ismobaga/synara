#pragma once

#include "synara/nn/module.hpp"

namespace synara
{

    class Optimizer
    {
    public:
        explicit Optimizer(std::vector<Tensor *> params) : params_(std::move(params)) {}
        virtual ~Optimizer() = default;
        virtual void step() = 0;
        virtual double learning_rate() const noexcept = 0;
        virtual void set_learning_rate(double lr) noexcept = 0;
        virtual void zero_grad()
        {
            for (Tensor *param : params_)
            {
                if (param != nullptr)
                    param->zero_grad();
            }
        }
        const std::vector<Tensor *> &parameters() const noexcept { return params_; }

    protected:
        std::vector<Tensor *> params_;
    };

} // namespace synara
