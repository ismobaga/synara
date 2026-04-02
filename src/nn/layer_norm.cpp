#include "synara/nn/layer_norm.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        class LayerNormNode : public Node
        {
        public:
            LayerNormNode(Tensor input,
                          Tensor normalized,
                          Tensor gamma,
                          Tensor beta,
                          bool affine,
                          Tensor::value_type eps,
                          std::vector<Tensor::value_type> inv_std)
                : input_(std::move(input)),
                  normalized_(std::move(normalized)),
                  gamma_(std::move(gamma)),
                  beta_(std::move(beta)),
                  affine_(affine),
                  eps_(eps),
                  inv_std_(std::move(inv_std))
            {
            }

            void backward(const Tensor &grad_output) override
            {
                const Size batch = grad_output.shape()[0];
                const Size features = grad_output.shape()[1];

                if (affine_ && gamma_.requires_grad())
                {
                    Tensor grad_gamma = Tensor::zeros(gamma_.shape(), false);
                    for (Size f = 0; f < features; ++f)
                    {
                        Tensor::value_type acc = 0.0;
                        for (Size n = 0; n < batch; ++n)
                        {
                            acc += grad_output.at({n, f}) * normalized_.at({n, f});
                        }
                        grad_gamma.at({0, f}) = acc;
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
                    for (Size f = 0; f < features; ++f)
                    {
                        Tensor::value_type acc = 0.0;
                        for (Size n = 0; n < batch; ++n)
                        {
                            acc += grad_output.at({n, f});
                        }
                        grad_beta.at({0, f}) = acc;
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
                for (Size n = 0; n < batch; ++n)
                {
                    Tensor::value_type sum_dy = 0.0;
                    Tensor::value_type sum_dy_xhat = 0.0;

                    for (Size f = 0; f < features; ++f)
                    {
                        const Tensor::value_type gamma_value = affine_ ? gamma_.at({0, f}) : 1.0;
                        const Tensor::value_type dxhat = grad_output.at({n, f}) * gamma_value;
                        sum_dy += dxhat;
                        sum_dy_xhat += dxhat * normalized_.at({n, f});
                    }

                    for (Size f = 0; f < features; ++f)
                    {
                        const Tensor::value_type gamma_value = affine_ ? gamma_.at({0, f}) : 1.0;
                        const Tensor::value_type dxhat = grad_output.at({n, f}) * gamma_value;
                        const Tensor::value_type xhat = normalized_.at({n, f});
                        const Tensor::value_type scaled =
                            static_cast<Tensor::value_type>(features) * dxhat - sum_dy - xhat * sum_dy_xhat;
                        grad_input.at({n, f}) =
                            inv_std_[n] * scaled / static_cast<Tensor::value_type>(features);
                    }
                }

                input_.accumulate_grad(grad_input);
                if (input_.grad_fn())
                {
                    input_.grad_fn()->backward(grad_input);
                }
            }

        private:
            Tensor input_;
            Tensor normalized_;
            Tensor gamma_;
            Tensor beta_;
            bool affine_;
            Tensor::value_type eps_;
            std::vector<Tensor::value_type> inv_std_;
        };

    } // namespace

    LayerNorm::LayerNorm(Size num_features, bool affine, Tensor::value_type eps)
        : num_features_(num_features),
          affine_(affine),
          eps_(eps),
          gamma_(Parameter(Tensor::ones(Shape({1, num_features}), true))),
          beta_(Parameter(Tensor::zeros(Shape({1, num_features}), true)))
    {
    }

    Tensor LayerNorm::forward(const Tensor &input)
    {
        if (input.rank() != 2)
        {
            throw ShapeError("LayerNorm::forward(): input must be rank 2.");
        }
        if (input.shape()[1] != num_features_)
        {
            throw ShapeError("LayerNorm::forward(): input feature dimension mismatch.");
        }

        const Size batch = input.shape()[0];
        const Size features = input.shape()[1];
        if (features == 0)
        {
            throw ShapeError("LayerNorm::forward(): number of features must be > 0.");
        }

        std::vector<Tensor::value_type> mean(batch, 0.0);
        std::vector<Tensor::value_type> inv_std(batch, 0.0);

        Tensor normalized = Tensor::zeros(input.shape(), false);
        const bool requires_grad =
            input.requires_grad() || (affine_ && (gamma_.requires_grad() || beta_.requires_grad()));
        Tensor output = Tensor::zeros(input.shape(), requires_grad);

        for (Size n = 0; n < batch; ++n)
        {
            for (Size f = 0; f < features; ++f)
            {
                mean[n] += input.at({n, f});
            }
            mean[n] /= static_cast<Tensor::value_type>(features);

            Tensor::value_type var = 0.0;
            for (Size f = 0; f < features; ++f)
            {
                const Tensor::value_type centered = input.at({n, f}) - mean[n];
                var += centered * centered;
            }
            var /= static_cast<Tensor::value_type>(features);
            inv_std[n] = 1.0 / std::sqrt(var + eps_);

            for (Size f = 0; f < features; ++f)
            {
                const Tensor::value_type xhat = (input.at({n, f}) - mean[n]) * inv_std[n];
                normalized.at({n, f}) = xhat;

                Tensor::value_type out = xhat;
                if (affine_)
                {
                    out = gamma_.tensor().at({0, f}) * xhat + beta_.tensor().at({0, f});
                }
                output.at({n, f}) = out;
            }
        }

        output.set_requires_grad(requires_grad);
        output.set_leaf(!requires_grad);
        if (requires_grad)
        {
            output.set_grad_fn(std::make_shared<LayerNormNode>(
                input,
                normalized,
                gamma_.tensor(),
                beta_.tensor(),
                affine_,
                eps_,
                inv_std));
        }

        return output;
    }

    std::vector<Parameter *> LayerNorm::parameters()
    {
        if (!affine_)
        {
            return {};
        }
        return {&gamma_, &beta_};
    }

    StateDict LayerNorm::state_dict(const std::string &prefix) const
    {
        StateDict out;
        if (affine_)
        {
            out.emplace(prefix + "weight", Tensor::from_vector(gamma_.tensor().shape(), std::vector<Tensor::value_type>(gamma_.tensor().data(), gamma_.tensor().data() + gamma_.tensor().numel()), false));
            out.emplace(prefix + "bias", Tensor::from_vector(beta_.tensor().shape(), std::vector<Tensor::value_type>(beta_.tensor().data(), beta_.tensor().data() + beta_.tensor().numel()), false));
        }
        return out;
    }

    void LayerNorm::load_state_dict(const StateDict &state, const std::string &prefix)
    {
        if (!affine_)
        {
            return;
        }

        const std::string weight_key = prefix + "weight";
        const auto w_it = state.find(weight_key);
        if (w_it == state.end())
        {
            throw ValueError("LayerNorm::load_state_dict(): missing key '" + weight_key + "'.");
        }
        if (w_it->second.shape() != gamma_.tensor().shape())
        {
            throw ShapeError("LayerNorm::load_state_dict(): shape mismatch for key '" + weight_key + "'.");
        }
        for (Size i = 0; i < gamma_.tensor().numel(); ++i)
        {
            gamma_.tensor().data()[i] = w_it->second.data()[i];
        }

        const std::string bias_key = prefix + "bias";
        const auto b_it = state.find(bias_key);
        if (b_it == state.end())
        {
            throw ValueError("LayerNorm::load_state_dict(): missing key '" + bias_key + "'.");
        }
        if (b_it->second.shape() != beta_.tensor().shape())
        {
            throw ShapeError("LayerNorm::load_state_dict(): shape mismatch for key '" + bias_key + "'.");
        }
        for (Size i = 0; i < beta_.tensor().numel(); ++i)
        {
            beta_.tensor().data()[i] = b_it->second.data()[i];
        }
    }

    Size LayerNorm::num_features() const noexcept
    {
        return num_features_;
    }

    bool LayerNorm::affine() const noexcept
    {
        return affine_;
    }

    Parameter &LayerNorm::weight() noexcept
    {
        return gamma_;
    }

    const Parameter &LayerNorm::weight() const noexcept
    {
        return gamma_;
    }

    Parameter &LayerNorm::bias() noexcept
    {
        return beta_;
    }

    const Parameter &LayerNorm::bias() const noexcept
    {
        return beta_;
    }

} // namespace synara