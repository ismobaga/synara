#include "synara/optim/clip_grad.hpp"

#include <cmath>
#include <stdexcept>

namespace synara
{

    float clip_grad_norm(const std::vector<Tensor *> &params, float max_norm)
    {
        if (max_norm <= 0.0f)
            throw std::invalid_argument("clip_grad_norm: max_norm must be positive");

        double total_sq = 0.0;
        for (Tensor *param : params)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
                continue;
            const Tensor &g = param->grad();
            for (std::size_t i = 0; i < g.numel(); ++i)
            {
                const double v = static_cast<double>(g.data()[i]);
                total_sq += v * v;
            }
        }

        const float total_norm = static_cast<float>(std::sqrt(total_sq));

        if (total_norm > max_norm)
        {
            const float scale = max_norm / total_norm;
            for (Tensor *param : params)
            {
                if (param == nullptr || !param->requires_grad() || !param->has_grad())
                    continue;
                Tensor &g = param->grad();
                for (std::size_t i = 0; i < g.numel(); ++i)
                    g.data()[i] *= scale;
            }
        }

        return total_norm;
    }

} // namespace synara
