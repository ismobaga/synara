#include "synara/nn/linear.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

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

                if (input_.requires_grad())
                {
                    Tensor grad_input = Tensor::zeros(input_.shape(), false);
                    for (Size n = 0; n < batch; ++n)
                    {
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

                    input_.accumulate_grad(grad_input);
                    if (input_.grad_fn())
                    {
                        input_.grad_fn()->backward(grad_input);
                    }
                }

                if (weight_.requires_grad())
                {
                    Tensor grad_weight = Tensor::zeros(weight_.shape(), false);
                    for (Size o = 0; o < out_features; ++o)
                    {
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

                    weight_.accumulate_grad(grad_weight);
                    if (weight_.grad_fn())
                    {
                        weight_.grad_fn()->backward(grad_weight);
                    }
                }

                if (use_bias_ && bias_.requires_grad())
                {
                    Tensor grad_bias = Tensor::zeros(bias_.shape(), false);
                    for (Size n = 0; n < batch; ++n)
                    {
                        for (Size o = 0; o < out_features; ++o)
                        {
                            grad_bias.at({0, o}) += grad_output.at({n, o});
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

        Tensor output = Tensor::zeros(
            Shape({input.shape()[0], out_features_}),
            input.requires_grad() || weight_.requires_grad() || (use_bias_ && bias_.requires_grad()));

        for (Size n = 0; n < input.shape()[0]; ++n)
        {
            for (Size o = 0; o < out_features_; ++o)
            {
                Tensor::value_type acc = 0.0f;
                for (Size i = 0; i < in_features_; ++i)
                {
                    acc += input.at({n, i}) * weight_.tensor().at({o, i});
                }
                if (use_bias_)
                {
                    acc += bias_.tensor().at({0, o});
                }
                output.at({n, o}) = acc;
            }
        }

        const bool requires_grad =
            input.requires_grad() || weight_.requires_grad() || (use_bias_ && bias_.requires_grad());
        output.set_requires_grad(requires_grad);
        output.set_leaf(!requires_grad);
        if (requires_grad)
        {
            output.set_grad_fn(std::make_shared<LinearNode>(input, weight_.tensor(), bias_.tensor(), use_bias_));
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
