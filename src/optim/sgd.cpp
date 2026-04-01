
#include "synara/optim/sgd.hpp"

#include <cmath>

#include "synara/core/error.hpp"
namespace synara
{
    namespace
    {
        void validate_options(const SGDOptions &options)
        {
            if (options.lr <= 0.0)
            {
                throw ValueError("SGD: learning rate must be positive");
            }
            if (options.momentum < 0.0 || options.momentum >= 1.0)
            {
                throw ValueError("SGD: momentum must be in [0, 1)");
            }
            if (options.weight_decay < 0.0)
            {
                throw ValueError("SGD: weight decay must be non-negative");
            }
            if (options.max_grad_norm < 0.0)
            {
                throw ValueError("SGD: max_grad_norm must be non-negative");
            }
        }
    } // namespace

    SGD::SGD(std::vector<Tensor *> params, double lr)
        : SGD(std::move(params), SGDOptions{lr}) {}

    SGD::SGD(std::vector<Tensor *> params, SGDOptions options)
        : Optimizer(std::move(params)), options_(options)
    {
        validate_options(options_);
    }

    void SGD::step()
    {
        for (Tensor *param : params_)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
            {
                continue;
            }
            if (param->grad().shape() != param->shape())
            {
                throw ShapeError("SGD: gradient shape does not match parameter shape");
            }

            Tensor &grad = param->grad();
            Tensor update = Tensor::zeros(param->shape(), false);

            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                Tensor::value_type g = grad.data()[i];
                if (options_.weight_decay > 0.0)
                {
                    g += static_cast<Tensor::value_type>(options_.weight_decay) * param->data()[i];
                }
                update.data()[i] = g;
            }

            if (options_.max_grad_norm > 0.0)
            {
                double sq_norm = 0.0;
                for (std::size_t i = 0; i < update.numel(); ++i)
                {
                    const double v = static_cast<double>(update.data()[i]);
                    sq_norm += v * v;
                }

                const double norm = std::sqrt(sq_norm);
                if (norm > options_.max_grad_norm && norm > 0.0)
                {
                    const double scale = options_.max_grad_norm / norm;
                    for (std::size_t i = 0; i < update.numel(); ++i)
                    {
                        update.data()[i] = static_cast<Tensor::value_type>(update.data()[i] * scale);
                    }
                }
            }

            if (options_.momentum > 0.0)
            {
                auto it = velocity_.find(param);
                if (it == velocity_.end())
                {
                    it = velocity_.emplace(param, Tensor::zeros(param->shape(), false)).first;
                }

                Tensor &vel = it->second;
                if (vel.shape() != param->shape())
                {
                    vel = Tensor::zeros(param->shape(), false);
                }

                for (std::size_t i = 0; i < param->numel(); ++i)
                {
                    vel.data()[i] = static_cast<Tensor::value_type>(options_.momentum) * vel.data()[i] + update.data()[i];
                    update.data()[i] = vel.data()[i];
                }
            }

            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                param->data()[i] -= static_cast<Tensor::value_type>(options_.lr) * update.data()[i];
            }
        }
    }

} // namespace synara