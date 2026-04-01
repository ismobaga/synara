
#include "synara/optim/sgd.hpp"
#include "synara/core/error.hpp"
namespace synara
{
    SGD::SGD(std::vector<Tensor *> params, double lr) : Optimizer(std::move(params)), lr_(lr)
    {
        if (lr_ <= 0.0)
        {
            throw ValueError("SGD: learning rate must be positive");
        }
    }

    void SGD::step()
    {
        for (Tensor *param : params_)
        {
            if (param == nullptr || !param->requires_grad() || !param->has_grad())
            {
                continue;
            }
            if (param->grad().shape() != param->shape())
            {
                throw ShapeError("SGD: gradient shape does not match parameter shape");
            }
            Tensor &grad = param->grad();
            for (std::size_t i = 0; i < param->numel(); ++i)
            {
                param->data()[i] -= lr_ * grad.data()[i];
            }
        }
    }

} // namespace synara