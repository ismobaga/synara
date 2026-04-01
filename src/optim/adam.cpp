#include "synara/optim/adam.hpp"

#include <cmath>

#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {
        void validate_adam_options(const AdamOptions &options)
        {
            if (options.lr <= 0.0)
            {
                throw ValueError("Adam: learning rate must be positive");
            }
            if (options.beta1 < 0.0 || options.beta1 >= 1.0)
            {
                throw ValueError("Adam: beta1 must be in [0, 1)");
            }
            if (options.beta2 < 0.0 || options.beta2 >= 1.0)
            {
                throw ValueError("Adam: beta2 must be in [0, 1)");
            }
            if (options.eps <= 0.0)
            {
                throw ValueError("Adam: eps must be positive");
            }
            if (options.weight_decay < 0.0)
            {
                throw ValueError("Adam: weight decay must be non-negative");
            }
        }
    } // namespace

    Adam::Adam(std::vector<Tensor *> params, double lr)
        : Adam(std::move(params), AdamOptions{lr}) {}

    Adam::Adam(std::vector<Tensor *> params, AdamOptions options)
        : Optimizer(std::move(params)), options_(options), step_count_(0)
    {
        validate_adam_options(options_);
    }

    void Adam::step()
    {
        step_count_++;

        const double bias_correction1 = 1.0 - std::pow(options_.beta1, step_count_);
        const double bias_correction2 = 1.0 - std::pow(options_.beta2, step_count_);

        for (Tensor *param : params_)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
            {
                continue;
            }
            if (param->grad().shape() != param->shape())
            {
                throw ShapeError("Adam: gradient shape does not match parameter shape");
            }

            Tensor &grad = param->grad();

            auto m_it = m_.find(param);
            if (m_it == m_.end())
            {
                m_it = m_.emplace(param, Tensor::zeros(param->shape(), false)).first;
            }

            auto v_it = v_.find(param);
            if (v_it == v_.end())
            {
                v_it = v_.emplace(param, Tensor::zeros(param->shape(), false)).first;
            }

            Tensor &m = m_it->second;
            Tensor &v = v_it->second;

            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                Tensor::value_type g = grad.data()[i];

                if (options_.weight_decay > 0.0)
                {
                    g += static_cast<Tensor::value_type>(options_.weight_decay) * param->data()[i];
                }

                m.data()[i] = static_cast<Tensor::value_type>(options_.beta1) * m.data()[i] +
                              (1.0f - static_cast<Tensor::value_type>(options_.beta1)) * g;

                Tensor::value_type g_sq = g * g;
                v.data()[i] = static_cast<Tensor::value_type>(options_.beta2) * v.data()[i] +
                              (1.0f - static_cast<Tensor::value_type>(options_.beta2)) * g_sq;

                Tensor::value_type m_hat = m.data()[i] / static_cast<Tensor::value_type>(bias_correction1);
                Tensor::value_type v_hat = v.data()[i] / static_cast<Tensor::value_type>(bias_correction2);

                param->data()[i] -= static_cast<Tensor::value_type>(options_.lr) * m_hat /
                                    (std::sqrt(v_hat) + static_cast<Tensor::value_type>(options_.eps));
            }
        }
    }

} // namespace synara
