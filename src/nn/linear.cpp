#include "synara/nn/linear.hpp"

#include <memory>
#include <stdexcept>

#include "synara/core/parallel.hpp"
#include <string>
#include <utility>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        bool should_parallelize_linear(Size batch, Size out_features, Size in_features)
        {
            return static_cast<long long>(batch) *
                       static_cast<long long>(out_features) *
                       static_cast<long long>(in_features) >=
                   static_cast<long long>(parallel_config().linear_threshold);
        }

        class LinearNode : public Node
        {
        public:
            LinearNode(Tensor input, Tensor weight, Tensor bias, bool use_bias)
                : input_(std::move(input)),
                  weight_(std::move(weight)),
                  bias_(std::move(bias)),
                  use_bias_(use_bias) {}

            void backward(const Tensor &grad_output) override
            {
                const Size batch = grad_output.shape()[0];
                const Size out_features = grad_output.shape()[1];
                const Size in_features = input_.shape()[1];
                const bool fast_path =
                    grad_output.is_contiguous() && input_.is_contiguous() && weight_.is_contiguous() &&
                    (!use_bias_ || bias_.is_contiguous());
                const bool parallel = should_parallelize_linear(batch, out_features, in_features);

                if (input_.requires_grad())
                {
                    Tensor grad_input = Tensor::zeros(input_.shape(), false);

                    if (fast_path)
                    {
                        const Tensor::value_type *go = grad_output.data();
                        const Tensor::value_type *w = weight_.data();
                        Tensor::value_type *gi = grad_input.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long sample = 0; sample < static_cast<long long>(batch); ++sample)
                        {
                            const Size n = static_cast<Size>(sample);
                            const Tensor::value_type *go_row = go + n * out_features;
                            Tensor::value_type *gi_row = gi + n * in_features;

                            for (Size o = 0; o < out_features; ++o)
                            {
                                const Tensor::value_type g = go_row[o];
                                const Tensor::value_type *w_row = w + o * in_features;
                                for (Size i = 0; i < in_features; ++i)
                                {
                                    gi_row[i] += g * w_row[i];
                                }
                            }
                        }
                    }
                    else
                    {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long sample = 0; sample < static_cast<long long>(batch); ++sample)
                        {
                            const Size n = static_cast<Size>(sample);
                            for (Size i = 0; i < in_features; ++i)
                            {
                                Tensor::value_type acc = 0.0f;
                                for (Size o = 0; o < out_features; ++o)
                                {
                                    acc += grad_output.at({n, o}) * weight_.at({o, i});
                                }
                                grad_input.at({n, i}) = acc;
                            }
                        }
                    }

                    input_.accumulate_grad(grad_input);
                    if (input_.grad_fn())
                    {
                        input_.grad_fn()->backward(grad_input);
                    }
                }

                if (weight_.requires_grad())
                {
                    Tensor grad_weight = Tensor::zeros(weight_.shape(), false);

                    if (fast_path)
                    {
                        const Tensor::value_type *go = grad_output.data();
                        const Tensor::value_type *x = input_.data();
                        Tensor::value_type *gw = grad_weight.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long out_index = 0; out_index < static_cast<long long>(out_features); ++out_index)
                        {
                            const Size o = static_cast<Size>(out_index);
                            Tensor::value_type *gw_row = gw + o * in_features;
                            for (Size n = 0; n < batch; ++n)
                            {
                                const Tensor::value_type g = go[n * out_features + o];
                                const Tensor::value_type *x_row = x + n * in_features;
                                for (Size i = 0; i < in_features; ++i)
                                {
                                    gw_row[i] += g * x_row[i];
                                }
                            }
                        }
                    }
                    else
                    {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long out_index = 0; out_index < static_cast<long long>(out_features); ++out_index)
                        {
                            const Size o = static_cast<Size>(out_index);
                            for (Size i = 0; i < in_features; ++i)
                            {
                                Tensor::value_type acc = 0.0f;
                                for (Size n = 0; n < batch; ++n)
                                {
                                    acc += grad_output.at({n, o}) * input_.at({n, i});
                                }
                                grad_weight.at({o, i}) = acc;
                            }
                        }
                    }

                    weight_.accumulate_grad(grad_weight);
                    if (weight_.grad_fn())
                    {
                        weight_.grad_fn()->backward(grad_weight);
                    }
                }

                if (use_bias_ && bias_.requires_grad())
                {
                    Tensor grad_bias = Tensor::zeros(bias_.shape(), false);
                    Tensor::value_type *gb = grad_bias.data();

                    if (fast_path)
                    {
                        const Tensor::value_type *go = grad_output.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long out_index = 0; out_index < static_cast<long long>(out_features); ++out_index)
                        {
                            const Size o = static_cast<Size>(out_index);
                            Tensor::value_type acc = 0.0f;
                            for (Size n = 0; n < batch; ++n)
                            {
                                acc += go[n * out_features + o];
                            }
                            gb[o] = acc;
                        }
                    }
                    else
                    {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
                        for (long long out_index = 0; out_index < static_cast<long long>(out_features); ++out_index)
                        {
                            const Size o = static_cast<Size>(out_index);
                            Tensor::value_type acc = 0.0f;
                            for (Size n = 0; n < batch; ++n)
                            {
                                acc += grad_output.at({n, o});
                            }
                            grad_bias.at({0, o}) = acc;
                        }
                    }

                    bias_.accumulate_grad(grad_bias);
                    if (bias_.grad_fn())
                    {
                        bias_.grad_fn()->backward(grad_bias);
                    }
                }
            }

        private:
            Tensor input_;
            Tensor weight_;
            Tensor bias_;
            bool use_bias_;
        };

    } // namespace

    Linear::Linear(Size in_features, Size out_features, bool use_bias)
        : in_features_(in_features),
          out_features_(out_features),
          use_bias_(use_bias),
          weight_(make_weight(in_features, out_features)),
          bias_(make_bias(out_features)) {}

    Tensor Linear::forward(const Tensor &input)
    {
        if (input.rank() != 2)
        {
            throw ShapeError("Linear::forward(): input must be rank 2.");
        }
        if (input.shape()[1] != in_features_)
        {
            throw ShapeError("Linear::forward(): input feature dimension mismatch.");
        }

        const Tensor &weight_tensor = weight_.tensor();
        const Tensor &bias_tensor = bias_.tensor();
        const bool requires_grad =
            input.requires_grad() || weight_.requires_grad() || (use_bias_ && bias_.requires_grad());
        const bool parallel = should_parallelize_linear(input.shape()[0], out_features_, in_features_);

        Tensor output = Tensor::zeros(Shape({input.shape()[0], out_features_}), requires_grad);

        if (input.is_contiguous() && weight_tensor.is_contiguous() && (!use_bias_ || bias_tensor.is_contiguous()))
        {
            const Tensor::value_type *x = input.data();
            const Tensor::value_type *w = weight_tensor.data();
            const Tensor::value_type *b = use_bias_ ? bias_tensor.data() : nullptr;
            Tensor::value_type *y = output.data();

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long sample = 0; sample < static_cast<long long>(input.shape()[0]); ++sample)
            {
                const Size n = static_cast<Size>(sample);
                const Tensor::value_type *x_row = x + n * in_features_;
                Tensor::value_type *y_row = y + n * out_features_;

                for (Size o = 0; o < out_features_; ++o)
                {
                    const Tensor::value_type *w_row = w + o * in_features_;
                    Tensor::value_type acc = use_bias_ ? b[o] : 0.0f;

                    Size i = 0;
                    for (; i + 3 < in_features_; i += 4)
                    {
                        acc += x_row[i] * w_row[i];
                        acc += x_row[i + 1] * w_row[i + 1];
                        acc += x_row[i + 2] * w_row[i + 2];
                        acc += x_row[i + 3] * w_row[i + 3];
                    }
                    for (; i < in_features_; ++i)
                    {
                        acc += x_row[i] * w_row[i];
                    }

                    y_row[o] = acc;
                }
            }
        }
        else
        {
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long sample = 0; sample < static_cast<long long>(input.shape()[0]); ++sample)
            {
                const Size n = static_cast<Size>(sample);
                for (Size o = 0; o < out_features_; ++o)
                {
                    Tensor::value_type acc = 0.0f;
                    for (Size i = 0; i < in_features_; ++i)
                    {
                        acc += input.at({n, i}) * weight_tensor.at({o, i});
                    }
                    if (use_bias_)
                    {
                        acc += bias_tensor.at({0, o});
                    }
                    output.at({n, o}) = acc;
                }
            }
        }

        output.set_requires_grad(requires_grad);
        output.set_leaf(!requires_grad);
        if (requires_grad)
        {
            output.set_grad_fn(std::make_shared<LinearNode>(input, weight_tensor, bias_tensor, use_bias_));
        }

        return output;
    }

    std::vector<Parameter *> Linear::parameters()
    {
        if (use_bias_)
        {
            return {&weight_, &bias_};
        }
        return {&weight_};
    }

    StateDict Linear::state_dict(const std::string &prefix) const
    {
        StateDict out;
        out.emplace(prefix + "weight", Tensor::from_vector(weight_.tensor().shape(), std::vector<Tensor::value_type>(weight_.tensor().data(), weight_.tensor().data() + weight_.tensor().numel()), false));
        if (use_bias_)
        {
            out.emplace(prefix + "bias", Tensor::from_vector(bias_.tensor().shape(), std::vector<Tensor::value_type>(bias_.tensor().data(), bias_.tensor().data() + bias_.tensor().numel()), false));
        }
        return out;
    }

    void Linear::load_state_dict(const StateDict &state, const std::string &prefix)
    {
        const std::string weight_key = prefix + "weight";
        const auto w_it = state.find(weight_key);
        if (w_it == state.end())
        {
            throw ValueError("Linear::load_state_dict(): missing key '" + weight_key + "'.");
        }
        if (w_it->second.shape() != weight_.tensor().shape())
        {
            throw ShapeError("Linear::load_state_dict(): shape mismatch for key '" + weight_key + "'.");
        }

        for (Size i = 0; i < weight_.tensor().numel(); ++i)
        {
            weight_.tensor().data()[i] = w_it->second.data()[i];
        }

        if (!use_bias_)
        {
            return;
        }

        const std::string bias_key = prefix + "bias";
        const auto b_it = state.find(bias_key);
        if (b_it == state.end())
        {
            throw ValueError("Linear::load_state_dict(): missing key '" + bias_key + "'.");
        }
        if (b_it->second.shape() != bias_.tensor().shape())
        {
            throw ShapeError("Linear::load_state_dict(): shape mismatch for key '" + bias_key + "'.");
        }

        for (Size i = 0; i < bias_.tensor().numel(); ++i)
        {
            bias_.tensor().data()[i] = b_it->second.data()[i];
        }
    }

    Parameter &Linear::weight() noexcept { return weight_; }

    const Parameter &Linear::weight() const noexcept { return weight_; }

    Parameter &Linear::bias() noexcept { return bias_; }

    const Parameter &Linear::bias() const noexcept { return bias_; }

    Size Linear::in_features() const noexcept { return in_features_; }

    Size Linear::out_features() const noexcept { return out_features_; }

    bool Linear::has_bias() const noexcept { return use_bias_; }

    Parameter Linear::make_weight(Size in_features, Size out_features) const
    {
        Tensor tensor = Tensor::zeros(Shape({out_features, in_features}), true);
        for (Size out = 0; out < out_features; ++out)
        {
            for (Size in = 0; in < in_features; ++in)
            {
                const long long centered = static_cast<long long>((out * in_features + in) % 7) - 3;
                tensor.at({out, in}) = static_cast<Tensor::value_type>(0.1 * centered);
            }
        }
        return Parameter(std::move(tensor));
    }

    Parameter Linear::make_bias(Size out_features) const
    {
        return Parameter(Tensor::zeros(Shape({1, out_features}), true));
    }

} // namespace synara
