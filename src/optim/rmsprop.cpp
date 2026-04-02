#include "synara/optim/rmsprop.hpp"

#include <cmath>

#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {
        void validate_rmsprop_options(const RMSpropOptions &options)
        {
            if (options.lr <= 0.0)
                throw ValueError("RMSprop: learning rate must be positive");
            if (options.alpha <= 0.0 || options.alpha >= 1.0)
                throw ValueError("RMSprop: alpha must be in (0, 1)");
            if (options.eps <= 0.0)
                throw ValueError("RMSprop: eps must be positive");
            if (options.weight_decay < 0.0)
                throw ValueError("RMSprop: weight decay must be non-negative");
            if (options.momentum < 0.0)
                throw ValueError("RMSprop: momentum must be non-negative");
        }
    } // namespace

    RMSprop::RMSprop(std::vector<Tensor *> params, double lr)
        : RMSprop(std::move(params), RMSpropOptions{lr}) {}

    RMSprop::RMSprop(std::vector<Tensor *> params, RMSpropOptions options)
        : Optimizer(std::move(params)), options_(options)
    {
        validate_rmsprop_options(options_);
    }

    void RMSprop::step()
    {
        for (Tensor *param : params_)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
                continue;

            if (param->grad().shape() != param->shape())
                throw ShapeError("RMSprop: gradient shape does not match parameter shape");

            Tensor &grad = param->grad();

            auto sq_it = square_avg_.find(param);
            if (sq_it == square_avg_.end())
                sq_it = square_avg_.emplace(param, Tensor::zeros(param->shape(), false)).first;
            Tensor &sq_avg = sq_it->second;

            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                Tensor::value_type g = grad.data()[i];

                if (options_.weight_decay > 0.0)
                    g += static_cast<Tensor::value_type>(options_.weight_decay) * param->data()[i];

                sq_avg.data()[i] =
                    static_cast<Tensor::value_type>(options_.alpha) * sq_avg.data()[i] +
                    (1.0f - static_cast<Tensor::value_type>(options_.alpha)) * g * g;

                const Tensor::value_type step_size =
                    g / (std::sqrt(sq_avg.data()[i]) + static_cast<Tensor::value_type>(options_.eps));

                if (options_.momentum > 0.0)
                {
                    auto buf_it = momentum_buf_.find(param);
                    if (buf_it == momentum_buf_.end())
                        buf_it = momentum_buf_.emplace(param, Tensor::zeros(param->shape(), false)).first;
                    Tensor &buf = buf_it->second;

                    buf.data()[i] =
                        static_cast<Tensor::value_type>(options_.momentum) * buf.data()[i] + step_size;
                    param->data()[i] -= static_cast<Tensor::value_type>(options_.lr) * buf.data()[i];
                }
                else
                {
                    param->data()[i] -= static_cast<Tensor::value_type>(options_.lr) * step_size;
                }
            }
        }
    }

} // namespace synara
