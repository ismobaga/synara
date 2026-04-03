#include "synara/optim/grad_clip.hpp"

#include <algorithm>
#include <cmath>

#include "synara/core/error.hpp"

namespace synara
{

    double clip_grad_norm_(const std::vector<Tensor *> &parameters, double max_norm)
    {
        if (max_norm <= 0.0)
        {
            throw ValueError("clip_grad_norm_: max_norm must be > 0.");
        }

        double sq_norm = 0.0;
        for (Tensor *p : parameters)
        {
            if (p == nullptr || !p->has_grad())
            {
                continue;
            }
            const Tensor &g = p->grad();
            for (Size i = 0; i < g.numel(); ++i)
            {
                const double v = static_cast<double>(g.data()[i]);
                sq_norm += v * v;
            }
        }

        const double total_norm = std::sqrt(sq_norm);
        if (total_norm <= max_norm || total_norm == 0.0)
        {
            return total_norm;
        }

        const double scale = max_norm / total_norm;
        for (Tensor *p : parameters)
        {
            if (p == nullptr || !p->has_grad())
            {
                continue;
            }
            Tensor &g = p->grad();
            for (Size i = 0; i < g.numel(); ++i)
            {
                g.data()[i] = static_cast<Tensor::value_type>(g.data()[i] * scale);
            }
        }

        return total_norm;
    }

    void clip_grad_value_(const std::vector<Tensor *> &parameters, double max_value)
    {
        if (max_value < 0.0)
        {
            throw ValueError("clip_grad_value_: max_value must be >= 0.");
        }

        const Tensor::value_type hi = static_cast<Tensor::value_type>(max_value);
        const Tensor::value_type lo = static_cast<Tensor::value_type>(-max_value);
        for (Tensor *p : parameters)
        {
            if (p == nullptr || !p->has_grad())
            {
                continue;
            }
            Tensor &g = p->grad();
            for (Size i = 0; i < g.numel(); ++i)
            {
                g.data()[i] = std::max(lo, std::min(hi, g.data()[i]));
            }
        }
    }

} // namespace synara
