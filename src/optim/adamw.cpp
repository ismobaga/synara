#include "synara/optim/adamw.hpp"

#include <cmath>

#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {
        void validate_adamw_options(const AdamWOptions &options)
        {
            if (options.lr <= 0.0)
                throw ValueError("AdamW: learning rate must be positive");
            if (options.beta1 < 0.0 || options.beta1 >= 1.0)
                throw ValueError("AdamW: beta1 must be in [0, 1)");
            if (options.beta2 < 0.0 || options.beta2 >= 1.0)
                throw ValueError("AdamW: beta2 must be in [0, 1)");
            if (options.eps <= 0.0)
                throw ValueError("AdamW: eps must be positive");
            if (options.weight_decay < 0.0)
                throw ValueError("AdamW: weight decay must be non-negative");
        }
    } // namespace

    AdamW::AdamW(std::vector<Tensor *> params, double lr)
        : AdamW(std::move(params), AdamWOptions{lr}) {}

    AdamW::AdamW(std::vector<Tensor *> params, AdamWOptions options)
        : Optimizer(std::move(params)), options_(options), step_count_(0)
    {
        validate_adamw_options(options_);
    }

    void AdamW::step()
    {
        step_count_++;

        const double bias_correction1 = 1.0 - std::pow(options_.beta1, step_count_);
        const double bias_correction2 = 1.0 - std::pow(options_.beta2, step_count_);

        for (Tensor *param : params_)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
                continue;

            if (param->grad().shape() != param->shape())
                throw ShapeError("AdamW: gradient shape does not match parameter shape");

            // Decoupled weight decay: applied directly to the parameter before
            // the adaptive gradient step.
            if (options_.weight_decay > 0.0)
            {
                const Tensor::value_type decay =
                    1.0f - static_cast<Tensor::value_type>(options_.lr * options_.weight_decay);
                for (std::size_t i = 0; i < param->numel(); ++i)
                    param->data()[i] *= decay;
            }

            Tensor &grad = param->grad();

            auto m_it = m_.find(param);
            if (m_it == m_.end())
                m_it = m_.emplace(param, Tensor::zeros(param->shape(), false)).first;

            auto v_it = v_.find(param);
            if (v_it == v_.end())
                v_it = v_.emplace(param, Tensor::zeros(param->shape(), false)).first;

            Tensor &m = m_it->second;
            Tensor &v = v_it->second;

            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                const Tensor::value_type g = grad.data()[i];

                m.data()[i] = static_cast<Tensor::value_type>(options_.beta1) * m.data()[i] +
                              (1.0f - static_cast<Tensor::value_type>(options_.beta1)) * g;

                v.data()[i] = static_cast<Tensor::value_type>(options_.beta2) * v.data()[i] +
                              (1.0f - static_cast<Tensor::value_type>(options_.beta2)) * g * g;

                const Tensor::value_type m_hat =
                    m.data()[i] / static_cast<Tensor::value_type>(bias_correction1);
                const Tensor::value_type v_hat =
                    v.data()[i] / static_cast<Tensor::value_type>(bias_correction2);

                param->data()[i] -=
                    static_cast<Tensor::value_type>(options_.lr) * m_hat /
                    (std::sqrt(v_hat) + static_cast<Tensor::value_type>(options_.eps));
            }
        }
    }

} // namespace synara
