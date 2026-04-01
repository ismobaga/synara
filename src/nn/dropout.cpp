#include "synara/nn/dropout.hpp"

#include <memory>

#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        class DropoutNode : public Node
        {
        public:
            DropoutNode(Tensor input, Tensor mask, Tensor::value_type scale)
                : input_(std::move(input)), mask_(std::move(mask)), scale_(scale)
            {
            }

            void backward(const Tensor &grad_output) override
            {
                if (!input_.requires_grad())
                {
                    return;
                }

                Tensor grad_input = Tensor::zeros(input_.shape(), false);
                for (Size i = 0; i < input_.numel(); ++i)
                {
                    grad_input.data()[i] = grad_output.data()[i] * mask_.data()[i] * scale_;
                }

                input_.accumulate_grad(grad_input);
                if (input_.grad_fn())
                {
                    input_.grad_fn()->backward(grad_input);
                }
            }

        private:
            Tensor input_;
            Tensor mask_;
            Tensor::value_type scale_;
        };

    } // namespace

    Dropout::Dropout(Tensor::value_type p, std::uint64_t seed)
        : p_(p), training_(true), rng_state_(seed)
    {
        if (p_ < 0.0f || p_ > 1.0f)
        {
            throw ValueError("Dropout::Dropout(): p must be in [0, 1].");
        }
    }

    Tensor Dropout::forward(const Tensor &input)
    {
        const bool requires_grad = input.requires_grad();
        Tensor output = Tensor::zeros(input.shape(), requires_grad);

        if (!training_ || p_ == 0.0f)
        {
            for (Size i = 0; i < input.numel(); ++i)
            {
                output.data()[i] = input.data()[i];
            }

            output.set_leaf(!requires_grad);
            output.set_requires_grad(requires_grad);
            if (requires_grad)
            {
                Tensor ones = Tensor::ones(input.shape(), false);
                output.set_grad_fn(std::make_shared<DropoutNode>(input, ones, 1.0f));
            }
            return output;
        }

        const Tensor::value_type keep_prob = 1.0f - p_;
        const Tensor::value_type scale = (keep_prob > 0.0f) ? (1.0f / keep_prob) : 0.0f;

        Tensor mask = Tensor::zeros(input.shape(), false);
        for (Size i = 0; i < input.numel(); ++i)
        {
            const bool keep = (p_ < 1.0f) ? (next_uniform01() >= p_) : false;
            mask.data()[i] = keep ? 1.0f : 0.0f;
            output.data()[i] = input.data()[i] * mask.data()[i] * scale;
        }

        output.set_leaf(!requires_grad);
        output.set_requires_grad(requires_grad);
        if (requires_grad)
        {
            output.set_grad_fn(std::make_shared<DropoutNode>(input, mask, scale));
        }

        return output;
    }

    void Dropout::train() noexcept
    {
        training_ = true;
    }

    void Dropout::eval() noexcept
    {
        training_ = false;
    }

    bool Dropout::is_training() const noexcept
    {
        return training_;
    }

    Tensor::value_type Dropout::p() const noexcept
    {
        return p_;
    }

    void Dropout::set_p(Tensor::value_type p)
    {
        if (p < 0.0f || p > 1.0f)
        {
            throw ValueError("Dropout::set_p(): p must be in [0, 1].");
        }
        p_ = p;
    }

    std::uint64_t Dropout::seed() const noexcept
    {
        return rng_state_;
    }

    void Dropout::set_seed(std::uint64_t seed) noexcept
    {
        rng_state_ = seed;
    }

    Tensor::value_type Dropout::next_uniform01()
    {
        // Deterministic LCG for reproducible tests.
        rng_state_ = rng_state_ * 6364136223846793005ULL + 1ULL;
        const std::uint32_t hi = static_cast<std::uint32_t>(rng_state_ >> 32);
        return static_cast<Tensor::value_type>(hi) / 4294967295.0f;
    }

} // namespace synara
