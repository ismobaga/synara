#include "synara/ops/math.hpp"

#include <cmath>

#include "synara/autograd/no_grad.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

namespace synara
{

    namespace
    {
        void require_contiguous(const Tensor &a, const char *op)
        {
            if (!a.is_contiguous())
            {
                throw ShapeError(std::string(op) + "(): requires contiguous tensor.");
            }
        }

        template <typename Fn>
        Tensor unary_apply(const Tensor &a, Fn fn)
        {
            Tensor out = Tensor::zeros(a.shape(), false);
            for (Size i = 0; i < a.numel(); ++i)
            {
                out.data()[i] = fn(a.data()[i]);
            }
            return out;
        }

    } // namespace

    Tensor exp(const Tensor &a)
    {
        require_contiguous(a, "exp");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::exp(x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<ExpNode>(a, out));
        }
        return out;
    }

    Tensor log(const Tensor &a)
    {
        require_contiguous(a, "log");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::log(x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<LogNode>(a));
        }
        return out;
    }

    Tensor log2(const Tensor &a)
    {
        require_contiguous(a, "log2");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::log2(x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<Log2Node>(a));
        }
        return out;
    }

    Tensor sqrt(const Tensor &a)
    {
        require_contiguous(a, "sqrt");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::sqrt(x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<SqrtNode>(a, out));
        }
        return out;
    }

    Tensor pow(const Tensor &a, Tensor::value_type exponent)
    {
        require_contiguous(a, "pow");
        Tensor out = unary_apply(a, [exponent](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::pow(x, exponent)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<PowNode>(a, exponent));
        }
        return out;
    }

    Tensor abs(const Tensor &a)
    {
        require_contiguous(a, "abs");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return static_cast<Tensor::value_type>(std::abs(x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<AbsNode>(a));
        }
        return out;
    }

    Tensor sign(const Tensor &a)
    {
        require_contiguous(a, "sign");
        Tensor out = unary_apply(a, [](Tensor::value_type x)
                                 { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<SignNode>(a));
        }
        return out;
    }

    Tensor clamp(const Tensor &a, Tensor::value_type min_val, Tensor::value_type max_val)
    {
        if (min_val > max_val)
        {
            throw ValueError("clamp(): min must be <= max.");
        }
        require_contiguous(a, "clamp");

        Tensor out = unary_apply(a, [min_val, max_val](Tensor::value_type x)
                                 { return std::max(min_val, std::min(max_val, x)); });

        const bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<ClampNode>(a, min_val, max_val));
        }
        return out;
    }

} // namespace synara
