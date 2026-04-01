#include "synara/nn/batch_norm.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        class BatchNormNode : public Node
        {
        public:
            BatchNormNode(Tensor input,
                          Tensor normalized,
                          Tensor gamma,
                          bool affine,
                          bool training,
                          Tensor::value_type eps,
                          std::vector<Tensor::value_type> inv_std)
                : input_(std::move(input)),
                  normalized_(std::move(normalized)),
                  gamma_(std::move(gamma)),
                  affine_(affine),
                  training_(training),
                  eps_(eps),
                  inv_std_(std::move(inv_std))
            {
            }

            void backward(const Tensor &grad_output) override
            {
                const Size batch = grad_output.shape()[0];
                const Size channels = grad_output.shape()[1];

                if (affine_ && gamma_.requires_grad())
                {
                    Tensor grad_gamma = Tensor::zeros(gamma_.shape(), false);
                    for (Size c = 0; c < channels; ++c)
                    {
                        Tensor::value_type acc = 0.0f;
                        for (Size n = 0; n < batch; ++n)
                        {
                            acc += grad_output.at({n, c}) * normalized_.at({n, c});
                        }
                        grad_gamma.at({0, c}) = acc;
                    }

                    gamma_.accumulate_grad(grad_gamma);
                    if (gamma_.grad_fn())
                    {
                        gamma_.grad_fn()->backward(grad_gamma);
                    }
                }

                if (affine_ && beta_.requires_grad())
                {
                    Tensor grad_beta = Tensor::zeros(beta_.shape(), false);
                    for (Size c = 0; c < channels; ++c)
                    {
                        Tensor::value_type acc = 0.0f;
                        for (Size n = 0; n < batch; ++n)
                        {
                            acc += grad_output.at({n, c});
                        }
                        grad_beta.at({0, c}) = acc;
                    }

                    beta_.accumulate_grad(grad_beta);
                    if (beta_.grad_fn())
                    {
                        beta_.grad_fn()->backward(grad_beta);
                    }
                }

                if (!input_.requires_grad())
                {
                    return;
                }

                Tensor grad_input = Tensor::zeros(input_.shape(), false);
                for (Size c = 0; c < channels; ++c)
                {
                    const Tensor::value_type gamma_value = affine_ ? gamma_.at({0, c}) : 1.0f;

                    if (!training_)
                    {
                        for (Size n = 0; n < batch; ++n)
                        {
                            grad_input.at({n, c}) = grad_output.at({n, c}) * gamma_value * inv_std_[c];
                        }
                        continue;
                    }

                    Tensor::value_type sum_dy = 0.0f;
                    Tensor::value_type sum_dy_xhat = 0.0f;
                    for (Size n = 0; n < batch; ++n)
                    {
                        const Tensor::value_type dy = grad_output.at({n, c}) * gamma_value;
                        sum_dy += dy;
                        sum_dy_xhat += dy * normalized_.at({n, c});
                    }

                    for (Size n = 0; n < batch; ++n)
                    {
                        const Tensor::value_type dy = grad_output.at({n, c}) * gamma_value;
                        const Tensor::value_type xhat = normalized_.at({n, c});
                        const Tensor::value_type scaled =
                            (static_cast<Tensor::value_type>(batch) * dy - sum_dy - xhat * sum_dy_xhat);
                        grad_input.at({n, c}) =
                            inv_std_[c] * scaled / static_cast<Tensor::value_type>(batch);
                    }
                }

                input_.accumulate_grad(grad_input);
                if (input_.grad_fn())
                {
                    input_.grad_fn()->backward(grad_input);
                }
            }

            void set_beta(Tensor beta)
            {
                beta_ = std::move(beta);
            }

        private:
            Tensor input_;
            Tensor normalized_;
            Tensor gamma_;
            Tensor beta_;
            bool affine_;
            bool training_;
            Tensor::value_type eps_;
            std::vector<Tensor::value_type> inv_std_;
        };

    } // namespace

    BatchNorm1d::BatchNorm1d(Size num_features, bool affine, Tensor::value_type eps, Tensor::value_type momentum)
        : num_features_(num_features),
          affine_(affine),
          eps_(eps),
          momentum_(momentum),
          training_(true),
          gamma_(Parameter(Tensor::ones(Shape({1, num_features}), true))),
          beta_(Parameter(Tensor::zeros(Shape({1, num_features}), true))),
          running_mean_(Tensor::zeros(Shape({1, num_features}), false)),
          running_var_(Tensor::ones(Shape({1, num_features}), false))
    {
    }

    Tensor BatchNorm1d::forward(const Tensor &input)
    {
        if (input.rank() != 2)
        {
            throw ShapeError("BatchNorm1d::forward(): input must be rank 2.");
        }
        if (input.shape()[1] != num_features_)
        {
            throw ShapeError("BatchNorm1d::forward(): input feature dimension mismatch.");
        }

        const Size batch = input.shape()[0];
        if (batch == 0)
        {
            throw ShapeError("BatchNorm1d::forward(): batch size must be > 0.");
        }

        std::vector<Tensor::value_type> mean(num_features_, 0.0f);
        std::vector<Tensor::value_type> var(num_features_, 0.0f);
        std::vector<Tensor::value_type> inv_std(num_features_, 0.0f);

        if (training_)
        {
            for (Size c = 0; c < num_features_; ++c)
            {
                for (Size n = 0; n < batch; ++n)
                {
                    mean[c] += input.at({n, c});
                }
                mean[c] /= static_cast<Tensor::value_type>(batch);

                for (Size n = 0; n < batch; ++n)
                {
                    const Tensor::value_type centered = input.at({n, c}) - mean[c];
                    var[c] += centered * centered;
                }
                var[c] /= static_cast<Tensor::value_type>(batch);

                inv_std[c] = 1.0f / std::sqrt(var[c] + eps_);

                running_mean_.at({0, c}) =
                    (1.0f - momentum_) * running_mean_.at({0, c}) + momentum_ * mean[c];
                running_var_.at({0, c}) =
                    (1.0f - momentum_) * running_var_.at({0, c}) + momentum_ * var[c];
            }
        }
        else
        {
            for (Size c = 0; c < num_features_; ++c)
            {
                mean[c] = running_mean_.at({0, c});
                var[c] = running_var_.at({0, c});
                inv_std[c] = 1.0f / std::sqrt(var[c] + eps_);
            }
        }

        const bool requires_grad =
            input.requires_grad() || (affine_ && (gamma_.requires_grad() || beta_.requires_grad()));

        Tensor normalized = Tensor::zeros(input.shape(), false);
        Tensor output = Tensor::zeros(input.shape(), requires_grad);

        for (Size n = 0; n < batch; ++n)
        {
            for (Size c = 0; c < num_features_; ++c)
            {
                const Tensor::value_type xhat = (input.at({n, c}) - mean[c]) * inv_std[c];
                normalized.at({n, c}) = xhat;

                Tensor::value_type out = xhat;
                if (affine_)
                {
                    out = gamma_.tensor().at({0, c}) * xhat + beta_.tensor().at({0, c});
                }
                output.at({n, c}) = out;
            }
        }

        output.set_requires_grad(requires_grad);
        output.set_leaf(!requires_grad);
        if (requires_grad)
        {
            auto node = std::make_shared<BatchNormNode>(
                input,
                normalized,
                gamma_.tensor(),
                affine_,
                training_,
                eps_,
                inv_std);
            node->set_beta(beta_.tensor());
            output.set_grad_fn(node);
        }

        return output;
    }

    std::vector<Parameter *> BatchNorm1d::parameters()
    {
        if (!affine_)
        {
            return {};
        }
        return {&gamma_, &beta_};
    }

    StateDict BatchNorm1d::state_dict(const std::string &prefix) const
    {
        StateDict out;

        if (affine_)
        {
            out.emplace(prefix + "weight", Tensor::from_vector(gamma_.tensor().shape(), std::vector<Tensor::value_type>(gamma_.tensor().data(), gamma_.tensor().data() + gamma_.tensor().numel()), false));
            out.emplace(prefix + "bias", Tensor::from_vector(beta_.tensor().shape(), std::vector<Tensor::value_type>(beta_.tensor().data(), beta_.tensor().data() + beta_.tensor().numel()), false));
        }

        out.emplace(prefix + "running_mean", Tensor::from_vector(running_mean_.shape(), std::vector<Tensor::value_type>(running_mean_.data(), running_mean_.data() + running_mean_.numel()), false));
        out.emplace(prefix + "running_var", Tensor::from_vector(running_var_.shape(), std::vector<Tensor::value_type>(running_var_.data(), running_var_.data() + running_var_.numel()), false));

        return out;
    }

    void BatchNorm1d::load_state_dict(const StateDict &state, const std::string &prefix)
    {
        if (affine_)
        {
            const std::string weight_key = prefix + "weight";
            const auto w_it = state.find(weight_key);
            if (w_it == state.end())
            {
                throw ValueError("BatchNorm1d::load_state_dict(): missing key '" + weight_key + "'.");
            }
            if (w_it->second.shape() != gamma_.tensor().shape())
            {
                throw ShapeError("BatchNorm1d::load_state_dict(): shape mismatch for key '" + weight_key + "'.");
            }
            for (Size i = 0; i < gamma_.tensor().numel(); ++i)
            {
                gamma_.tensor().data()[i] = w_it->second.data()[i];
            }

            const std::string bias_key = prefix + "bias";
            const auto b_it = state.find(bias_key);
            if (b_it == state.end())
            {
                throw ValueError("BatchNorm1d::load_state_dict(): missing key '" + bias_key + "'.");
            }
            if (b_it->second.shape() != beta_.tensor().shape())
            {
                throw ShapeError("BatchNorm1d::load_state_dict(): shape mismatch for key '" + bias_key + "'.");
            }
            for (Size i = 0; i < beta_.tensor().numel(); ++i)
            {
                beta_.tensor().data()[i] = b_it->second.data()[i];
            }
        }

        const std::string mean_key = prefix + "running_mean";
        const auto m_it = state.find(mean_key);
        if (m_it == state.end())
        {
            throw ValueError("BatchNorm1d::load_state_dict(): missing key '" + mean_key + "'.");
        }
        if (m_it->second.shape() != running_mean_.shape())
        {
            throw ShapeError("BatchNorm1d::load_state_dict(): shape mismatch for key '" + mean_key + "'.");
        }
        for (Size i = 0; i < running_mean_.numel(); ++i)
        {
            running_mean_.data()[i] = m_it->second.data()[i];
        }

        const std::string var_key = prefix + "running_var";
        const auto v_it = state.find(var_key);
        if (v_it == state.end())
        {
            throw ValueError("BatchNorm1d::load_state_dict(): missing key '" + var_key + "'.");
        }
        if (v_it->second.shape() != running_var_.shape())
        {
            throw ShapeError("BatchNorm1d::load_state_dict(): shape mismatch for key '" + var_key + "'.");
        }
        for (Size i = 0; i < running_var_.numel(); ++i)
        {
            running_var_.data()[i] = v_it->second.data()[i];
        }
    }

    void BatchNorm1d::train() noexcept
    {
        training_ = true;
    }

    void BatchNorm1d::eval() noexcept
    {
        training_ = false;
    }

    bool BatchNorm1d::is_training() const noexcept
    {
        return training_;
    }

    Size BatchNorm1d::num_features() const noexcept
    {
        return num_features_;
    }

    bool BatchNorm1d::affine() const noexcept
    {
        return affine_;
    }

    Parameter &BatchNorm1d::weight() noexcept
    {
        return gamma_;
    }

    const Parameter &BatchNorm1d::weight() const noexcept
    {
        return gamma_;
    }

    Parameter &BatchNorm1d::bias() noexcept
    {
        return beta_;
    }

    const Parameter &BatchNorm1d::bias() const noexcept
    {
        return beta_;
    }

    const Tensor &BatchNorm1d::running_mean() const noexcept
    {
        return running_mean_;
    }

    const Tensor &BatchNorm1d::running_var() const noexcept
    {
        return running_var_;
    }

} // namespace synara
